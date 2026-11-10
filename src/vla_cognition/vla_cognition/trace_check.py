"""Read-only consistency monitor for VLM traces (no ROS imports, never imported by gate_core.py)."""
import re
from collections import deque

GROUP_OF_CLASS = {
    'person': 'person',
    'cow': 'animal', 'dog': 'animal', 'cat': 'animal', 'horse': 'animal', 'sheep': 'animal', 'bird': 'animal',
    'bear': 'animal', 'elephant': 'animal', 'zebra': 'animal', 'giraffe': 'animal',
    'bicycle': 'vehicle', 'motorcycle': 'vehicle', 'bus': 'vehicle', 'truck': 'vehicle', 'train': 'vehicle', 'boat': 'vehicle',
}
WORD_GROUP = {}
for g, words in {
    'person': ['person', 'people', 'pedestrian', 'pedestrians', 'man', 'men', 'woman', 'women', 'child', 'boy', 'girl', 'human', 'walker'],
    'animal': ['cow', 'cows', 'cattle', 'bull', 'bulls', 'ox', 'calf', 'bovine', 'dog', 'dogs', 'cat', 'cats', 'horse', 'horses',
               'sheep', 'goat', 'goats', 'deer', 'bird', 'birds', 'animal', 'animals'],
    'vehicle': ['truck', 'lorry', 'bus', 'motorcycle', 'motorbike', 'bicycle', 'bike', 'cyclist', 'van'],
}.items():
    for w in words:
        WORD_GROUP[w] = g
NEGATIONS = {'no', 'not', 'without', 'nor', 'never', 'none', "n't", 'any'}


def groups_in_text(text):
    toks = re.findall(r"[a-z']+", text.lower())
    found = set()
    for i, tk in enumerate(toks):
        g = WORD_GROUP.get(tk)
        if g is None:
            continue
        if any(t in NEGATIONS or t.endswith("n't") for t in toks[max(0, i - 3):i]):
            continue
        found.add(g)
    return found


class TraceChecker:
    def __init__(self, min_conf=0.40, history_s=12.0, obstacle_h_min=0.04, clear_h_min=0.08, img_h=800.0):
        self.min_conf, self.history_s, self.img_h = min_conf, history_s, img_h
        self.obstacle_h_min, self.clear_h_min = obstacle_h_min, clear_h_min
        self.hist = deque()
        self.counts = {'n': 0, 'consistent': 0, 'empty': 0, 'unsupported_object': 0,
                       'unsupported_obstacle': 0, 'missed_obstacle': 0}

    def observe(self, now, dets, corridor_hfrac):
        groups = {GROUP_OF_CLASS.get(d.get('class'), 'other') for d in (dets or []) if d.get('confidence', 0.0) >= self.min_conf}
        any_h = max([d.get('height', 0.0) / self.img_h for d in (dets or []) if d.get('confidence', 0.0) >= self.min_conf], default=0.0)
        self.hist.append((now, groups, corridor_hfrac or 0.0, any_h))
        while self.hist and now - self.hist[0][0] > self.history_s:
            self.hist.popleft()

    def check(self, now, trace):
        text = str(trace.get('reasoning', '') or '').strip()
        t_llm = trace.get('t_llm_ms')
        win = (float(t_llm) / 1000.0 + 2.0) if isinstance(t_llm, (int, float)) and t_llm else 6.0
        recent = [h for h in self.hist if now - h[0] <= win]
        seen_groups = set().union(*[h[1] for h in recent]) if recent else set()
        best_h = max([h[3] for h in recent], default=0.0)
        latest_h = self.hist[-1][2] if self.hist else 0.0
        flags = []
        if not text:
            verdict = 'EMPTY'
            self.counts['empty'] += 1
        else:
            for g in sorted(groups_in_text(text)):
                if g not in seen_groups:
                    flags.append('UNSUPPORTED_OBJECT:%s' % g)
            claim = trace.get('obstacle_in_path')
            if claim is True and best_h < self.obstacle_h_min:
                flags.append('UNSUPPORTED_OBSTACLE')
            if claim is False and latest_h > self.clear_h_min:
                flags.append('MISSED_OBSTACLE')
            verdict = flags[0].split(':')[0] if flags else 'CONSISTENT'
            if any(f.startswith('UNSUPPORTED_OBJECT') for f in flags):
                self.counts['unsupported_object'] += 1
            if 'UNSUPPORTED_OBSTACLE' in flags:
                self.counts['unsupported_obstacle'] += 1
            if 'MISSED_OBSTACLE' in flags:
                self.counts['missed_obstacle'] += 1
            if not flags:
                self.counts['consistent'] += 1
        self.counts['n'] += 1
        return verdict, flags

    def summary(self):
        c = self.counts
        return 'n=%d consistent=%d empty=%d unsupported_object=%d unsupported_obstacle=%d missed_obstacle=%d' % (
            c['n'], c['consistent'], c['empty'], c['unsupported_object'], c['unsupported_obstacle'], c['missed_obstacle'])
