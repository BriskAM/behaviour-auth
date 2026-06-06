# main.py  (create this in the root, next to the behaveguard/ folder)
from behavguard.ui.enrollment import run_enrollment
from behavguard import pipeline

subject_id = "alice"
segment_events = run_enrollment(subject_id)
pipeline.enroll(subject_id, segment_events)
