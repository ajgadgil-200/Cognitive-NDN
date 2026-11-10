"""Pure-Python core of the safety gate (no ROS imports) so it can be unit-tested offline.

Inputs : detection lists (YOLO boxes in an 800x800 image) and the current time.
Outputs: (linear.x, angular.z) every control tick.
The VLM is NOT an input. Commands are a deterministic function of the detections and elapsed time.
"""
from dataclasses import dataclass

CRUISE = 'CRUISE'
AVOIDING = 'AVOIDING'
PASSING = 'PASSING'
RETURNING = 'RETURNING'
STOPPED = 'STOPPED'


@dataclass
class GateParams:
    # camera
    img_w: float = 800.0
    img_h: float = 800.0
    # detection hygiene
    min_conf: float = 0.40
    ego_w_frac: float = 0.90
    ego_bottom_frac: float = 0.95
    ego_max_conf: float = 0.60
    det_timeout: float = 1.0
    # corridor
    corridor_half: float = 0.60
    lat_limit_m: float = 2.0
    h_ref_m: float = 1.5
    w_ref_m: float = 1.7
    # proximity thresholds (box height / frame height)
    react_h: float = 0.08
    stop_h: float = 0.30
    stop_bottom_frac: float = 0.0
    avoid_enabled: bool = False
    slow_bottom_band: float = 0.12
    mem_hold: float = 1.5
    recover_rate: float = 0.5
    react_persist: int = 2
    # motion
    cruise_v: float = 1.6
    avoid_v: float = 1.0
    avoid_w: float = 0.3
    avoid_t: float = 2.0
    pass_t: float = 1.5
    return_t: float = 2.0
    cooldown: float = 3.0
    left_sign: float = 1.0
    # stop latch / lost-obstacle handling
    clear_hold: float = 1.0
    lost_close_h: float = 0.20
    lost_ahead_offset: float = 0.40
    lost_miss: int = 2
    lost_hold: float = 1.0


class GateCore:
    def __init__(self, params=None):
        self.p = params or GateParams()
        self.state = CRUISE
        self.state_start = 0.0
        self.turn_dir = 1.0
        self.cooldown_until = 0.0
        self.last_msg = None
        self.obs = None
        self.react_count = 0
        self.close_track = None
        self.lost_until = 0.0
        self.clear_since = None
        self.stop_reason = ''
        self.slowing = False
        self.f_out = 1.0
        self.mem_f = 1.0
        self.mem_until = 0.0
        self.last_t = None
        self._events = []

    def pop_events(self):
        ev, self._events = self._events, []
        return ev

    def _ev(self, msg):
        self._events.append(msg)

    def _set(self, new_state, now, why=''):
        self._ev('[STATE] %s -> %s%s' % (self.state, new_state, (' (%s)' % why) if why else ''))
        self.state = new_state
        self.state_start = now

    def _valid(self, d):
        p = self.p
        conf = d.get('confidence', 0.0)
        if conf < p.min_conf:
            return None
        w, h = d.get('width', 0.0), d.get('height', 0.0)
        x, y = d.get('x', p.img_w / 2.0), d.get('y', 0.0)
        if h <= 0.0:
            return None
        bottom = y + h / 2.0
        if w > p.ego_w_frac * p.img_w and bottom >= p.ego_bottom_frac * p.img_h and conf < p.ego_max_conf:
            return None
        cx = p.img_w / 2.0
        dx = x - cx
        offset = dx / cx
        if abs(offset) > p.corridor_half:
            return None
        size = max(h, (w * h) ** 0.5)
        scale = max(h / p.h_ref_m, w / p.w_ref_m)
        if abs(dx) / scale > p.lat_limit_m:
            return None
        return (d, offset, size / p.img_h, bottom / p.img_h)

    def on_detections(self, now, dets):
        p = self.p
        self.last_msg = now
        best = None
        for d in dets or []:
            o = self._valid(d)
            if o is not None and (best is None or o[2] > best[2]):
                best = o
        self.react_count = self.react_count + 1 if (best is not None and best[2] > p.react_h) else 0
        if best is not None:
            self.close_track = {'off': best[1], 'hf': best[2], 'miss': 0} if best[2] >= p.lost_close_h else None
        elif self.close_track is not None:
            self.close_track['miss'] += 1
            if self.close_track['miss'] >= p.lost_miss:
                if abs(self.close_track['off']) < p.lost_ahead_offset:
                    self.lost_until = now + p.lost_hold
                    self._ev('[OVERRIDE] lost-close obstacle (last height_frac=%.2f offset=%+.2f): hold stop %.1f s'
                             % (self.close_track['hf'], self.close_track['off'], p.lost_hold))
                self.close_track = None
        self.obs = best

    def step(self, now):
        p = self.p
        dt = 0.0 if self.last_t is None else max(0.0, now - self.last_t)
        self.last_t = now

        # 1. watchdog (highest priority): perception silent -> stop
        if self.last_msg is None or now - self.last_msg > p.det_timeout:
            if self.state != CRUISE:
                self._ev('[STATE] %s -> %s (watchdog reset)' % (self.state, CRUISE))
            self.state, self.clear_since = CRUISE, None
            self.stop_reason = 'no fresh detections - perception down'
            self.f_out = 0.0
            return 0.0, 0.0

        obs = self.obs
        emerg = obs is not None and (obs[2] > p.stop_h or (p.stop_bottom_frac > 0.0 and obs[3] >= p.stop_bottom_frac))

        # 2. emergency stop: valid in EVERY state, including mid-manoeuvre
        if emerg or now < self.lost_until:
            if emerg:
                self.stop_reason = '%s too close (height_frac=%.2f bottom=%.2f)' % (
                    obs[0].get('class'), obs[2], obs[3])
            else:
                self.stop_reason = 'lost-close hold'
            if self.state != STOPPED:
                self._set(STOPPED, now, self.stop_reason)
            self.clear_since = None
            return 0.0, 0.0

        # 3. latched stop: resume only after the corridor has been clear for clear_hold seconds
        if self.state == STOPPED:
            if obs is not None and obs[2] > p.react_h:
                self.clear_since = None
                self.stop_reason = 'obstacle still in corridor'
                return 0.0, 0.0
            if self.clear_since is None:
                self.clear_since = now
            if now - self.clear_since < p.clear_hold:
                self.stop_reason = 'clear-hold'
                return 0.0, 0.0
            self._set(CRUISE, now, 'corridor clear')
            self.cooldown_until = now
            self.f_out = 0.0
            self.mem_until = 0.0

        # 4. timed avoidance manoeuvre
        if self.state == AVOIDING:
            if now - self.state_start >= p.avoid_t:
                self._set(PASSING, now)
            else:
                return p.avoid_v, self.turn_dir * p.avoid_w
        if self.state == PASSING:
            if now - self.state_start >= p.pass_t:
                self._set(RETURNING, now)
            else:
                return p.avoid_v, 0.0
        if self.state == RETURNING:
            if now - self.state_start >= p.return_t:
                self._set(CRUISE, now)
                self.cooldown_until = now + p.cooldown
            else:
                return p.avoid_v, -self.turn_dir * p.avoid_w

        # 5a. slow-down zone (default)
        if not p.avoid_enabled:
            f = 1.0
            if obs is not None and obs[2] > p.react_h and self.react_count >= p.react_persist:
                ramp_end = 0.9 * p.stop_h
                f = min(f, max(0.0, (ramp_end - obs[2]) / (ramp_end - p.react_h)))
            if obs is not None and p.stop_bottom_frac > 0.0:
                f = min(f, max(0.0, (p.stop_bottom_frac - obs[3]) / p.slow_bottom_band))
            if f < 1.0 and obs is not None:
                self.mem_f, self.mem_until = f, now + p.mem_hold
            elif obs is None and now < self.mem_until:
                f = min(f, self.mem_f)
            if f > self.f_out:
                f = min(f, self.f_out + p.recover_rate * dt)
            self.f_out = f
            if f <= 0.02 and (obs is not None or now < self.mem_until):
                if self.state != STOPPED:
                    self._set(STOPPED, now, 'stopping size reached')
                self.stop_reason = 'stopping size reached'
                self.clear_since = None
                return 0.0, 0.0
            if f < 1.0:
                if not self.slowing and (obs is not None or now < self.mem_until):
                    self._ev('[SLOW] %s size_frac=%.2f bottom=%.2f -> speed factor %.2f'
                             % (obs[0].get('class') if obs is not None else 'dropout-hold',
                                obs[2] if obs is not None else 0.0, obs[3] if obs is not None else 0.0, f))
                self.slowing = True
                return p.cruise_v * f, 0.0
            self.slowing = False
            return p.cruise_v, 0.0

        # 5b. cruise: start a swerve if a persistent obstacle is in the corridor (avoid_enabled only)
        if (obs is not None and obs[2] > p.react_h and self.react_count >= p.react_persist
                and now >= self.cooldown_until):
            det, off, hf, bot = obs
            self.turn_dir = (1.0 if off >= 0.0 else -1.0) * p.left_sign
            self._ev('[AVOID] %s height_frac=%.2f offset=%+.2f -> turn_dir=%+.0f'
                     % (det.get('class'), hf, off, self.turn_dir))
            self._set(AVOIDING, now)
            return p.avoid_v, self.turn_dir * p.avoid_w
        return p.cruise_v, 0.0
