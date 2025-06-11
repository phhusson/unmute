import json
import random

from unmute.llm.system_prompt import (
    CONVERSATION_STARTER_SYSTEM_PROMPT_TEMPLATE,
    fill_system_prompt_template,
)
from unmute.tts.voices import VoiceList


def random_id():
    return "".join(random.choices("1234567890", k=8))


METATEMPLATE = "<bos><start_of_turn>user{system_prompt}\n\n\nHello!<end_of_turn>\n<start_of_turn>model\n"


def main():
    voice_list = VoiceList()
    personalities = [v.personality for v in voice_list.voices if v.personality]

    prompts = []

    for _ in range(10000):
        personality = random.choice(personalities)
        system_prompt = fill_system_prompt_template(
            CONVERSATION_STARTER_SYSTEM_PROMPT_TEMPLATE,
            system_prompt_style=personality + "\n" + random_id(),  # to break caching
        )
        full_prompt = METATEMPLATE.format(system_prompt=system_prompt)
        prompts.append(full_prompt)

    s = json.dumps(
        [
            {
                "id": random_id(),
                "conversations": [
                    {
                        "from": "human",
                        "value": full_prompt,
                    },
                    {
                        "from": "gpt",
                        "value": "Here are the main ideas of Jeff Walker's Product Launch Formula that can be applied by a growth marketing agency for their clients.",
                    },
                ],
            }
            for full_prompt in prompts
        ],
        indent=2,
    )

    # fill_system_prompt_template(
    #     CONVERSATION_STARTER_SYSTEM_PROMPT_TEMPLATE,
    #     system_prompt_style=self._instructions,
    # )
    print(s)


if __name__ == "__main__":
    main()
