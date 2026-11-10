"""SQA: pairwise comparison of stylization methods judged by a vision-language model (Google GenAI SDK).

Fill in API_KEY and BASELINE_PATH below, set RUN = True and run `python3 utils/sqa.py`. Every Hist2Style-vs-baseline comparison (both
orderings, randomized, up to 1000 per baseline) is sent to the judge once and cached in utils/sqa_results/; rerunning only tallies.
A method's SQA is the share of its comparisons with Hist2Style that it wins, a tie counting half (Hist2Style = 50 by definition).
Inputs must be sRGB (no embedded profile = sRGB); non-sRGB images are rejected before any API call.
BASELINE_PATH/<method>/c-c<i>_s-s<j>.jpg are the outputs of each method on the EvaluationSet pairs. CONTENT_PATH and STYLE_PATH default to the
EvaluationSet that eval.py downloads (content_set/ and style_set/ hold the content and style image of each pair under the same file name).
ADJUST THE THREAD COUNT TO YOUR API RATE LIMITS."""
import os
import dill
import numpy as np
from tqdm.contrib.concurrent import thread_map
from google import genai
from google.genai import types
import sys
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)  # repo root: run.py
from run import validate_srgb_images

RUN = False
API_KEY = None
CONTENT_PATH = os.path.join(REPO, "data", "EvaluationSet", "content_set")
STYLE_PATH = os.path.join(REPO, "data", "EvaluationSet", "style_set")
BASELINE_PATH = None
# The paper's judge. As of Nov 2026, Google no longer serves gemini-2.5-pro to new API users;
# gemini-3.1-pro-preview can run the same prompt.
MODEL = "gemini-2.5-pro"
DONE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sqa_results/")

BASELINE_NAMES = ["DLUT", "Hist2Style", "IDT", "salut", "WCT2_org", "xia_baseline_srgb", "PhotoWCT2"]


def evaluate_style_transfer(api_key, content_path, style_path, output_a_path, output_b_path, seed=42, model_id=MODEL):
	"""Ask the judge which of two outputs better stylizes the content image after the style image; returns the API response (or {"error": ...})."""
	client = genai.Client(api_key=api_key)

	def load_image_part(path):
		mime_type = "image/png" if path.lower().endswith(".png") else "image/jpeg"
		with open(path, "rb") as f:
			return types.Part.from_bytes(data=f.read(), mime_type=mime_type)

	try:
		part_content, part_style = load_image_part(content_path), load_image_part(style_path)
		part_out_a, part_out_b = load_image_part(output_a_path), load_image_part(output_b_path)
	except IOError as e:
		return {"error": f"Failed to load image files: {e}"}

	contents = [
		"You are an expert in photography, and computer vision, and photo analysis.",
		"Below are four images:",
		"\n1. CONTENT IMAGE (Structure reference):", part_content,
		"\n2. STYLE IMAGE (Aesthetic reference):", part_style,
		"\n3. OUTPUT <A>:", part_out_a,
		"\n4. OUTPUT <B>:", part_out_b,
		"\nTask: Compare Output <A> and Output <B>.",
		"Criteria: Identify the image that best preserves the structure and details of the content image, while matching the color palette, tone, and overall aesthetic of the style image without introducing artifacts.",
		"Response Format: First, provide 1 sentence of reasoning. Then, on a new line, output strictly either '<A>', '<B>', or '<T>' (tie).",
	]
	try:
		return client.models.generate_content(model=model_id, contents=contents, config=types.GenerateContentConfig(seed=seed))
	except Exception as e:
		return {"error": f"GenAI API Error: {e}"}


def main():
	os.makedirs(DONE_PATH, exist_ok=True)
	all_pairs = [f"c-c{j}_s-s{i}.jpg" for i in range(19) for j in range(136)]
	all_combos = []
	for p in all_pairs:
		for b in BASELINE_NAMES:
			if b != "Hist2Style":
				all_combos.append(f"{p}|{b}|Hist2Style")
				all_combos.append(f"{p}|Hist2Style|{b}")
	np.random.seed(42)
	all_combos = np.random.permutation(all_combos)
	done_pairs = os.listdir(DONE_PATH)
	all_combos = [c for c in all_combos if c not in done_pairs]
	todo_counts = {b: 1000 for b in BASELINE_NAMES}
	total_combinations = sum(v for k, v in todo_counts.items() if k != "Hist2Style")
	for c in done_pairs:
		_, b1, b2 = c.split("|")
		todo_counts[b1] -= 1
		todo_counts[b2] -= 1
	todo_combos = []
	for b in BASELINE_NAMES:
		if b != "Hist2Style" and todo_counts[b] > 0:
			todo_combos.extend([c for c in all_combos if b in c.split("|")[1:]][:todo_counts[b]])
	all_combos = [c for c in all_combos if c in todo_combos]
	print(f"Loaded from cache. There are {len(all_combos)} combinations to evaluate out of {total_combinations}")

	if RUN:
		validate_srgb_images(sorted({os.path.join(CONTENT_PATH, p) for p in all_pairs} | {os.path.join(STYLE_PATH, p) for p in all_pairs}
		                      | {os.path.join(BASELINE_PATH, b, p) for b in BASELINE_NAMES for p in all_pairs}))

		def evaluate_combo(combo):
			p, b, c = combo.split("|")
			cp, sp = os.path.join(CONTENT_PATH, p), os.path.join(STYLE_PATH, p)
			result = evaluate_style_transfer(API_KEY, cp, sp, os.path.join(BASELINE_PATH, b, p), os.path.join(BASELINE_PATH, c, p), seed=42)
			with open(os.path.join(DONE_PATH, combo), "wb") as f:
				dill.dump(result, f)

		thread_map(evaluate_combo, all_combos)

	bad_combos = []
	for combo in os.listdir(DONE_PATH):
		with open(os.path.join(DONE_PATH, combo), "rb") as f:
			out = dill.load(f)
		if isinstance(out, dict) and "error" in out:
			bad_combos.append(combo)
			print(out["error"])
	print(f"There are {len(bad_combos)} faulty combinations (removed; rerun to retry them)")
	for combo in bad_combos:
		os.remove(os.path.join(DONE_PATH, combo))

	results = {b: {"wins": 0, "losses": 0, "ties": 0, "total_valid": 0} for b in BASELINE_NAMES}
	combos = list(os.listdir(DONE_PATH))
	np.random.shuffle(combos)
	for combo in combos:
		with open(os.path.join(DONE_PATH, combo), "rb") as f:
			out = dill.load(f)
		if not hasattr(out, "text"):
			print(out)
			continue
		verdict = out.text[-3:]
		_, b, c = combo.split("|")
		if verdict == "<A>":
			results[b]["wins"] += 1; results[c]["losses"] += 1
		elif verdict == "<B>":
			results[b]["losses"] += 1; results[c]["wins"] += 1
		elif verdict == "<T>":
			results[b]["ties"] += 1; results[c]["ties"] += 1
		else:
			continue
		results[b]["total_valid"] += 1
		results[c]["total_valid"] += 1

	for b in BASELINE_NAMES:
		r = results[b]
		if b == "Hist2Style":
			print("Hist2Style: SQA 50 (the reference every method is compared with)")
		elif r["total_valid"] == 0:
			print(f"{b}: no valid comparisons yet")
		else:
			n = r["total_valid"]
			print(f"{b}: SQA {(r['wins'] + r['ties'] / 2) / n * 100:.2f}. {r['wins'] / n * 100:.2f}% wins, {r['losses'] / n * 100:.2f}% losses, {r['ties'] / n * 100:.2f}% ties out of {n} total valid")


if __name__ == "__main__":
	main()
