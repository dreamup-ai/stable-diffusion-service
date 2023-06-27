import math
import re


def make_enclosures_symmetrical(prompt, open_char="(", close_char=")"):
    """Make the number of open and close characters in the prompt symmetrical.
    A prompt may have many sets of enclosed subprompts, and the number of open and close characters
    may be unbalanced. This function makes the number of open and close characters symmetrical by
    removing open or close characters to the beginning or end of the enclosed subprompts.
    """
    new_prompt = prompt
    subprompt_finder = re.compile(f"({re.escape(open_char)}+)([^{re.escape(close_char)}]+)({re.escape(close_char)}+)", re.MULTILINE | re.IGNORECASE)
    subprompts = subprompt_finder.findall(prompt)
    for open_chars, subprompt, close_chars in subprompts:
        if len(open_chars) == len(close_chars):
            continue
        
        if len(open_chars) > len(close_chars):
            new_subprompt = (open_char * len(close_chars)) + subprompt + close_chars
        else:
            new_subprompt = open_chars + subprompt + (close_char * len(open_chars))
        new_prompt = new_prompt.replace(open_chars + subprompt + close_chars, new_subprompt)
    return new_prompt


# Copied without modification from nataili hopefully in accordance with their licensing terms. Thanks nataili!
def rewrite_a1111_style_weights(prompt):
    prompt = make_enclosures_symmetrical(prompt, "(", ")")
    prompt = make_enclosures_symmetrical(prompt, "[", "]")
    def rewrite_for_char(prompt, open_char="(", close_char=")", max_chars=5, weight_basis=1.1):
        # Iterate over the maximum number of modifier characters downwards
        for num_chars in range(max_chars, 0, -1):
            open = open_char * num_chars
            close = close_char * num_chars

            # Find subprompt with num_chars chars
            subprompt_open_i = prompt.find(open)
            subprompt_close_i = prompt.find(close, subprompt_open_i + 1)
            while subprompt_open_i != -1 and subprompt_close_i != -1:
                subprompt = prompt[subprompt_open_i + num_chars : subprompt_close_i]
                og_subprompt = subprompt
                weight = None

                # if subprompt contains a ":" use that weight as the base weight
                if subprompt.find(":") != -1 and num_chars > 1:
                    subprompt, weight = subprompt.split(":")
                    if not weight:
                        weight = 1 / weight_basis
                    else:
                        weight = float(weight) * math.pow(weight_basis, num_chars)

                # otherwise use the weight basis
                elif subprompt.find(":") == -1:
                    weight = math.pow(weight_basis, num_chars)

                elif subprompt.find(":") != -1 and num_chars == 1:
                    subprompt, weight = subprompt.split(":")
                    if not weight:
                        weight = 1
                    else:
                        weight = float(weight)
                    
                
                # Replace the prompt with the nataili-style prompt
                if weight is not None:
                    prompt = prompt.replace(open + og_subprompt + close, f"({subprompt}:{weight:.2f})")

                # Find next subprompt
                subprompt_open_i = prompt.find(open, subprompt_open_i + 1)
                subprompt_close_i = prompt.find(close, subprompt_open_i + 1)
        return prompt

    # Rewrite for ( and ) trains
    prompt = rewrite_for_char(prompt, open_char="(", close_char=")", max_chars=5, weight_basis=1.1)
    # Rewrite for [ and ] trains
    prompt = rewrite_for_char(prompt, open_char="[", close_char="]", max_chars=5, weight_basis=0.9)

    return prompt


nat_weights = re.compile("\((.+?):([\d\.]+)\)", re.MULTILINE | re.IGNORECASE)
def rewrite_prompt_for_compel(prompt):
    prompt = rewrite_a1111_style_weights(prompt)

    # convert the prompt weighting syntax from `a tall (man:1.1) picking apricots` to `a tall (man)1.1 picking apricots`
    prompt = nat_weights.sub(r"(\1)\2", prompt)
   
    
    return prompt


def get_prompt_embeds(compel, prompt):
    prompt = rewrite_prompt_for_compel(prompt)
    conditioning = compel.build_conditioning_tensor(prompt)
    return conditioning