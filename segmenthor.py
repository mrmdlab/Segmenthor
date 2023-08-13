import os

if not os.getenv("subprocess"):
    import json
    import requests
    from gui import SegmentThor
    import enums

    def verifyFail():
        with open("SUBSCRIPTION_NEEDED.LOG","w") as f:
            msg="This beta version has expired. Please contact fengh@imcb.a-star.edu.sg for subscription"
            f.write(msg)
            print(msg)
    def InternetFail():
        with open("INTERNET_FAIL.LOG","w") as f:
            msg="Please connect to the Internet before using this software"
            f.write(msg)
            print(msg)

    def downloadModel(model):
        model_pth=f"checkpoints/sam_{model}.pth"
        if not os.path.isfile(model_pth):
            model_url={
                "vit_b":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "vit_l":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_h":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            }
            file=requests.get(model_url[model])
            os.makedirs("checkpoints",exist_ok=True)
            with open(model_pth,"wb") as f:
                print(f"Downloading the model {model}")
                f.write(file.content)

    config={
        "model":"vit_b",
        "mask_path":"same"
    }
    try:
        with open("config.json") as f:
            cfg=f.read()
            cfg=json.loads(cfg)
            config.update(cfg)
    # in case the config file doesn't exist
    except:
        pass
    downloadModel(config["model"])

    try:
        url = 'https://www.fastmock.site/mock/58a16e152ae47a52c80240fb09bb6bf3/segment_thor/login'
        body = {'username': f'Segmenthor_{enums.VERSION}', 'password': '96k53m'}
        response = requests.post(url, data=body)
        success=False
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                success=True
                SegmentThor(config)
        if not success:
            verifyFail()
    except requests.exceptions.ConnectionError as e:
        InternetFail()