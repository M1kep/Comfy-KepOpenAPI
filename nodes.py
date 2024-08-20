from typing import Tuple

import torch
from openai import Client as OpenAIClient
import os
import json

from .lib import credentials, image


class ImageWithPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "file_paths_json": ("STRING", {}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Generate a high quality caption for the image. The most important aspects of the image should be described first. If needed, weights can be applied to the caption in the following format: '(word or phrase:weight)', where the weight should be a float less than 2.",
                    },
                ),
                "max_tokens": ("INT", {"min": 1, "max": 2048, "default": 77}),
                "give_up_after": ("INT", {"min": 1, "max": 128, "default": 8})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_completion"

    CATEGORY = "OpenAI"

    def __init__(self):
        self.open_ai_client: OpenAIClient = OpenAIClient(
            api_key=credentials.get_open_ai_api_key()
        )

    def generate_completion(
        self, images: torch.Tensor, file_paths_json: str, prompt: str, max_tokens: int, give_up_after: int
    ) -> str:
        file_paths = json.loads(file_paths_json)
        ret = []
        for path, img in zip(file_paths, images):
            img = torch.unsqueeze(img, 0)
            b64image = image.pil2base64(image.tensor2pil(img))
            response = self.open_ai_client.chat.completions.create(
                model="gpt-4o",
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64image}"},
                            },
                        ],
                    }
                ],
            )
            print(f"DEBUG [ImageWithPrompt] Response from OpenAI API for {path}: {response}")
            if len(response.choices) == 0:
                s = "No response from OpenAI API"
            else:
                s = response.choices[0].message.content
            print(f"DEBUG [ImageWithPrompt] Response from OpenAI API for {path}: {s}")
            ret.append(response.choices[0].message.content)

        fail_counts = self.handle_failure_and_success(file_paths, [True if 'True' in s else False for s in ret], give_up_after)

        for img, path, response, num_fails in zip(images, file_paths, ret, fail_counts):
            base_file_name = os.path.basename(path)
            if 'True' in response:
                path = f"output/good-{base_file_name}"
            else:
                path = f"output/bad-{base_file_name}_{num_fails:03d}"
            
            with open(f"{path}.png", "wb") as f:
                image.tensor2pil(img).save(f, "PNG")
            with open(f"{path}.txt", "w", encoding='utf-8') as f:
                f.write(response)
        
        return ("\n".join(ret), )

    def handle_failure_and_success(self, file_paths: list[str], are_good: list[str], GIVE_UP_AFTER: int = 10, FAIL_DIR: str = './failed') -> list[int]:
        fail_counts = []
        for file_path, is_good in zip(file_paths, are_good):
            if is_good:
                print(f"DEBUG success => deleting {file_path}", flush=True)
                os.remove(file_path)
                fail_counts.append(0)
            else:
                base_name = os.path.basename(file_path)
                dir_name = os.path.dirname(file_path)
                meta_file_path = os.path.join(dir_name, f".{base_name}.metadata")
                
                try:
                    with open(meta_file_path) as f:
                        metadata = json.load(f)
                    num_fails = metadata.get("num_fails", 0)
                except FileNotFoundError:
                    num_fails = 0
                
                num_fails += 1
                fail_counts.append(num_fails)
                if num_fails > GIVE_UP_AFTER:
                    os.makedirs(FAIL_DIR, exist_ok=True)
                    os.rename(file_path, os.path.join(FAIL_DIR, base_name))
                    try:
                        os.remove(meta_file_path)
                    except:
                        pass
                    print(f"INFO [Execute python (UNSAFE)] {file_path} moved to {FAIL_DIR} after {num_fails} failures")
                else:
                    metadata = {"num_fails": num_fails}
                    with open(meta_file_path, 'w') as f:
                        json.dump(metadata, f)
                    print(f"INFO [Execute python (UNSAFE)] failure count incremented to {num_fails} for {file_path}")
        return fail_counts
