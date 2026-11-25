"""Retain drawer references that passed a physical joint-command replay."""
import argparse
import json
import shutil
from pathlib import Path


def physical_success(data):
    return bool(data["drawer_pull_m"] > .08
                and data["cabinet_hand_contact_fraction"] > .35
                and data["drawer_hand_contact_fraction"] > .08
                and data["left_support_fraction"] > .8
                and data["right_support_fraction"] > .8
                and data["min_root_height_m"] > .6
                and data["max_root_tilt_rad"] < .4
                and data["first_fall_step"] is None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--screen-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    manifest=json.loads((a.source/"manifest.json").read_text())
    kept=[];screened=[]
    for clip in manifest["clips"]:
        name=clip["name"]
        if clip["split"]=="train":
            path=a.screen_dir/f"reference_screen_{name}.json"
        elif name=="val_00":
            path=a.screen_dir/"reference_joint_final_val00.json"
        elif name=="val_03":
            path=a.screen_dir/"reference_joint_final_val_03_revised.json"
        else:
            path=a.screen_dir/f"reference_joint_final_{name}.json"
        data=json.loads(path.read_text())
        passed=physical_success(data)
        screened.append({"name":name,"split":clip["split"],"passed":passed,
                         "drawer_pull_m":data["drawer_pull_m"],
                         "max_root_tilt_rad":data["max_root_tilt_rad"],
                         "first_fall_step":data["first_fall_step"],
                         "screen_file":str(path)})
        if passed:
            relative=Path(clip["file"])
            dest=a.output/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(a.source/relative,dest)
            kept.append(clip)
    manifest["clips"]=kept
    manifest["physical_screening"]=screened
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"kept {len(kept)} of {len(screened)} clips")

if __name__=="__main__":main()
