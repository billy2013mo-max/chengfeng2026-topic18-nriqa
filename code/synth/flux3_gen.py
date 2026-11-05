"""Pilot 2: synthesise over-generation with FLUX 3 Image Klein (image-conditioned).

The model re-renders the image from a caption + reference image, which is the
mechanism behind the real "过度生成" defect (AI enhancement / AI re-rendering).
The caption comes from the official prompt upsampler, prompted to change only
the texture and keep everything else identical.

Run with venv_flux/bin/python (diffusers >= 0.42, transformers >= 5).
"""
import argparse, csv, json, os, sys, time
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='models/flux3-klein')
ap.add_argument('--pup', default='models/flux3-pup')
ap.add_argument('--list', default='/tmp/synth/src.csv')
ap.add_argument('--out', default='/tmp/synth/flux3')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--steps', type=int, default=16)
ap.add_argument('--levels', default='mild,strong')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

INSTR = {
    'mild': ("Apply a restrained AI photo enhancement to this image: slightly smooth skin and "
             "fine texture, clean up micro detail, a faint plastic quality. Keep content, layout, "
             "text, colours and identity identical."),
    'strong': ("Apply a heavy AI enhancement and generation pipeline to this image: unnaturally "
               "smooth plastic skin, fine texture and natural grain gone, small details smeared or "
               "invented, over-smoothed edges. Keep the layout, composition, text and identity "
               "identical."),
}

from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: E402
from diffusers import ModularPipeline                                    # noqa: E402

t0 = time.time()
pup_model = Qwen3_5ForConditionalGeneration.from_pretrained(
    a.pup, dtype=torch.bfloat16, device_map='cuda').eval()
pup_proc = AutoProcessor.from_pretrained(a.pup)
SYSP = json.load(open(os.path.join(a.pup, 'system_prompts.json')))['i2i']
print(f'upsampler loaded {time.time()-t0:.0f}s', flush=True)

blocks = ModularPipeline.from_pretrained(a.model).blocks
blocks.sub_blocks.pop('prompt_upsampler', None)
pipe = blocks.init_pipeline(a.model)
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
pipe.to('cuda')
print(f'image model loaded {time.time()-t0:.0f}s', flush=True)


@torch.no_grad()
def caption(im, instr):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSP}]},
            {"role": "user", "content": [{"type": "image", "image": im},
                                         {"type": "text", "text": instr}]}]
    text = pup_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = pup_proc(text=[text], images=[im], return_tensors='pt').to('cuda')
    out = pup_model.generate(**inp, max_new_tokens=700, do_sample=False)
    return pup_proc.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()


names = [r['name'] for r in csv.DictReader(open(a.list))]
if a.limit:
    names = names[:a.limit]
levels = a.levels.split(',')

for lvl in levels:
    for i, n in enumerate(names):
        p = f'data/images/{n}'
        if not os.path.exists(p):
            p = f'data/images_supp/{n}'
        im = Image.open(p).convert('RGB')
        try:
            cap = caption(im, INSTR[lvl])
            out = pipe(prompt=cap, image=im, num_inference_steps=a.steps,
                       generator=torch.Generator(device='cpu').manual_seed(i),
                       output=['images', 'prompt'])
            out['images'][0].save(os.path.join(a.out, f'{lvl}_{n}'))
            open(os.path.join(a.out, f'{lvl}_{n}.caption.txt'), 'w').write(cap)
        except Exception as e:
            print('FAIL', n, type(e).__name__, str(e)[:200], flush=True)
        if (i + 1) % 10 == 0:
            print(f'{lvl} {i+1}/{len(names)}  {time.time()-t0:.0f}s', flush=True)
print('done', flush=True)
