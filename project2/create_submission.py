import os
import tarfile
import sys

def create_archive(output_filename="project2.tar.gz"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = ["bayesfilter.py", "report.pdf"]
    optional_files = ["pacmanagent.py"]
    
    missing_required = []
    for f in required_files:
        path = os.path.join(script_dir, f)
        if not os.path.isfile(path):
            missing_required.append(f)
            
    if missing_required:
        print(f"[ERROR] Missing required file(s): {', '.join(missing_required)}")
        if "report.pdf" in missing_required:
            print("[INFO] Please compile 'template-project2.tex' (e.g. on Overleaf) and save the resulting PDF as 'report.pdf' in this folder.")
        sys.exit(1)
        
    files_to_pack = [("bayesfilter.py", os.path.join(script_dir, "bayesfilter.py")),
                     ("report.pdf", os.path.join(script_dir, "report.pdf"))]
                     
    for opt in optional_files:
        opt_path = os.path.join(script_dir, opt)
        if os.path.isfile(opt_path):
            files_to_pack.append((opt, opt_path))
            print(f"[INFO] Included optional bonus file: {opt}")
            
    out_path = os.path.join(script_dir, output_filename)
    with tarfile.open(out_path, "w:gz") as tar:
        for arcname, file_path in files_to_pack:
            tar.add(file_path, arcname=arcname)
            print(f"[OK] Added: {arcname}")
            
    print(f"\n[SUCCESS] Created archive: {out_path}")
    print("[CONTENTS]:")
    with tarfile.open(out_path, "r:gz") as tar:
        for member in tar.getmembers():
            print(f" - {member.name} ({member.size} bytes)")

if __name__ == "__main__":
    archive_name = sys.argv[1] if len(sys.argv) > 1 else "project2.tar.gz"
    create_archive(archive_name)
