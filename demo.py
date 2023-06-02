import webuiapi

# create API client
api = webuiapi.WebUIApi()
# api = webuiapi.WebUIApi(host='10.25.10.165', port=7862)

# create API client with custom host, port and https
# api = webuiapi.WebUIApi(host='webui.example.com', port=443, use_https=True)

# create API client with default sampler, steps.
# api = webuiapi.WebUIApi(sampler='Euler a', steps=20)

# optionally set username, password when --api-auth is set on webui.
# api.set_auth('username', 'password')


from PIL import Image

image = Image.open("1.jpeg")
# mask = Image.open("下载.png")

prompt = "1girl, black shirt,((solo)), long hair, looking at viewer, realistic, jewelry, black eyes, lips, shirt, brown hair, closed mouth, earrings, forehead,black hair, simple background, upper body, watermark"
negative_prompt = "easynegative, canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render"

override_settings = {"sd_model_checkpoint": "chilloutmix_NiPrunedFp32Fix.safetensors"}

res = api.controlnet_detect([image], module="canny", processor_res=512, threshold_a=100, threshold_b=200)

from webuiapi.webuiapi import ControlNetUnit
controlnet_units = []
a = ControlNetUnit()
a.input_image = image
a.mask = res.image
a.module = "canny"
a.model = "control_sd15_canny [fef5e48e]"
a.threshold_a = 100
a.threshold_b = 200
a.processor_res = 512
a.guidance_start = 0.0
a.guidance_end = 1.0
a.resize_mode = "Crop and Resize"
controlnet_units.append(a)

alwayson_scripts = {}
# alwayson_scripts["Additional networks for generating"] = {"args": [True, False, 'LoRA', 'sun_5-000025(385432eaba02)', 0.5, 0.9]}
for i in range(100):
    result = api.img2img(prompt=prompt, negative_prompt=negative_prompt, images=[image], seed=-1, alwayson_scripts=alwayson_scripts, controlnet_units=controlnet_units)
    result.images[0].save(f"imgtoimgt{i}.png")

# result = api.txt2img(prompt=prompt, negative_prompt=negative_prompt, seed=555, alwayson_scripts={"Additional networks for generating":{"args": [True, False, 'LoRA', 'sun_5-000025(385432eaba02)', 0.9, 0.9]}}, override_settings=override_settings)
#
# result.images[0].save("texttoimg.png")

# api = webuiapi.WebUIApi(baseurl="http://127.0.0.1:7860")
#
# result = api.img2text(image=image)
#
# print(result.info)
# payload = {}


# Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 555, Size: 512x512, Model hash: fc2511737a, Model: chilloutmix_NiPrunedFp32Fix, Denoising strength: 0.75, Clip skip: 2, ControlNet 0: "preprocessor: canny, model: control_sd15_canny [fef5e48e], weight: 1, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: False, control mode: Balanced, preprocessor params: (512, 100, 200)"