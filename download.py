import requests

folderL = ['diffusion', 'divdamp', 'vi_rhow_solver', 'vert_adv_limiter', 'horiz_adv_flux', 'horiz_adv_limiter']

for folder in folderL:
    url = "http://r-ccs-climate.riken.jp/members/yashiro/IAB_database/snapshot."+folder+".pe000000"

    response = requests.get(url)

    open("dyn_"+folder+"/snapshot.dyn_"+folder+"_new.pe000000", "wb").write(response.content)
