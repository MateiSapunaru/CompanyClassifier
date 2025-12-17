import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [eda|train|audit] [-- additional args for subcommand]")
        print("\nCommands:")
        print("  eda    - Run exploratory data analysis")
        print("  train  - Train baseline model and generate predictions")
        print("  audit  - Create audit sample for manual review")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "eda":
        sys.argv = ["eda.py"] + sys.argv[2:]
        from scripts.eda import main as eda_main
        eda_main()
    elif command == "train":
        # Forwardează argumentele către scriptul de train (argparse citește sys.argv)
        sys.argv = ["train_predict_baseline.py"] + sys.argv[2:]
        from scripts.train_predict_baseline import main as train_main
        train_main()
    elif command == "audit":
        # Forwardează argumentele către scriptul de audit
        sys.argv = ["make_audit_sample.py"] + sys.argv[2:]
        from scripts.make_audit_sample import main as audit_main
        audit_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: eda, train, audit")
        sys.exit(1)

if __name__ == "__main__":
    main()