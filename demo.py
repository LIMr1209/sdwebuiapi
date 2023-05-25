import webuiapi

# create API client
api = webuiapi.WebUIApi()

# create API client with custom host, port
#api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)

# create API client with custom host, port and https
#api = webuiapi.WebUIApi(host='webui.example.com', port=443, use_https=True)

# create API client with default sampler, steps.
#api = webuiapi.WebUIApi(sampler='Euler a', steps=20)

# optionally set username, password when --api-auth is set on webui.
# api.set_auth('username', 'password')


from PIL import Image
image = Image.open("1.jpeg")

prompt = "1girl, ((solo)), long hair, looking at viewer, realistic, jewelry, black eyes, lips, shirt, brown hair, closed mouth, earrings, forehead, white shirt, black hair, simple background, upper body, watermark"
negative_prompt = "easynegative, canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render"


override_settings = {"sd_model_checkpoint":"chilloutmix_NiPrunedFp32Fix.safetensors"}

# result = api.img2img(prompt=prompt, negative_prompt=negative_prompt, images=[image], seed=555, alwayson_scripts={"Additional networks for generating":{"args": [True, False, 'LoRA', 'sun_5-000025(385432eaba02)', 0.5, 0.9]}}, override_settings=override_settings)
#
# result.images[0].save("imgtoimg.png")


# result = api.txt2img(prompt=prompt, negative_prompt=negative_prompt, seed=555, alwayson_scripts={"Additional networks for generating":{"args": [True, False, 'LoRA', 'sun_5-000025(385432eaba02)', 0.9, 0.9]}}, override_settings=override_settings)
#
# result.images[0].save("texttoimg.png")

api = webuiapi.WebUIApi(baseurl="http://127.0.0.1:7860")

result = api.img2text(image=image)

print(result.info)
