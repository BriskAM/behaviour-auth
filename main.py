import argparse
import sys
from behavguard.ui.enrollment import run_enrollment
from behavguard import pipeline

def main():
    parser = argparse.ArgumentParser(
        description="BehaveGuard — Behavioral Continuous Authentication via Keystroke Dynamics"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")
    
    # enroll command
    enroll_parser = subparsers.add_parser("enroll", help="Start enrollment session to train a model")
    enroll_parser.add_argument("--subject", default="alice", help="Subject ID (default: alice)")
    enroll_parser.add_argument(
        "--model", 
        choices=["lstm", "svm"], 
        default="lstm", 
        help="Model architecture to train (default: lstm)"
    )
    
    # score command
    score_parser = subparsers.add_parser("score", help="Start continuous live scoring session")
    score_parser.add_argument("--subject", default="alice", help="Subject ID (default: alice)")
    score_parser.add_argument(
        "--duration", 
        type=int, 
        default=300, 
        help="Duration of live scoring session in seconds (default: 300)"
    )
    
    args = parser.parse_args()
    
    if args.command == "enroll":
        segment_events = run_enrollment(args.subject)
        pipeline.enroll(args.subject, segment_events, model_type=args.model)
    elif args.command == "score":
        pipeline.score_live(args.subject, duration_seconds=args.duration)

if __name__ == "__main__":
    main()
