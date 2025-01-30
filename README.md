# ADG

Our algorithm is built on top of the `diffusers` package, providing a plug-and-play solution with a low learning curve for users. We have currently uploaded the algorithm code and the code for some of the key visualization results presented in the paper. The code for complete testing is undergoing privacy desensitization and will be released soon.

You can use the following Python code to run the proposed ADG algorithm:

```python
from diffusers import StableDiffusion3Pipeline
from method.ADG_SD3 import ADG_SD3

# Initialize the pipeline and integrate ADG
pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16)
setattr(pipeline, 'ADG', ADG_SD3.__get__(pipeline))

# Generate an image using ADG
prompt = 'what you want'
image = pipeline.ADG_SD3(prompt=prompt, num_inference_steps=10, guidance_scale=4, num_images_per_prompt=1).images[0]
image.save("output.jpg")
```

In the `vis.ipynb` file, we provide the code for reproducing some of the key visualizations from the paper, allowing you to explore and validate the experimental results.