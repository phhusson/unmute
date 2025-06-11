import argparse

from unmute.tts.voices import copy_voice_to_prod

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy voice to production server")
    parser.add_argument(
        "path_on_server",
        type=str,
        help="The path by which the voice is referred to by the TTS server "
        "(=relative to the voice directory)",
    )
    args = parser.parse_args()

    copy_voice_to_prod(args.path_on_server)
