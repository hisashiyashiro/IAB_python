import subprocess

folderL = ['diffusion', 'divdamp', 'vi_rhow_solver', 'vert_adv_limiter', 'horiz_adv_flux', 'horiz_adv_limiter']

for folder in folderL:
    print("\n##############################################################################")
    print("Starting kernel: dyn_"+folder)
    subprocess.run(["python", "dyn_"+folder+"/main.py"])
    print("##############################################################################\n")
