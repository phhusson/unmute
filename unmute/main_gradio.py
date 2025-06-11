import pprint
from typing import Any

import gradio as gr
import pandas as pd
import plotly.express as px
from fastrtc import Stream, get_hf_turn_credentials

from unmute.unmute_handler import GradioUpdate, UnmuteHandler

if __name__ == "__main__":
    gradio_chatbot = gr.Chatbot(type="messages")
    gradio_debug_textbox = gr.Textbox(label="Debug dict")
    gradio_debug_plot = gr.Plot(label="Debug plot")

    def update_outputs(
        _chatbot_state: Any,
        _debug_textbox_state: Any,
        _debug_plot_state: Any,
        update: GradioUpdate,
    ):
        # Not sure if this is expected behavior, but it seems necessary to send updates
        # to all of the components even if you don't want to change them. Otherwise they
        # get overwritten.
        chatbot_state = update.chat_history
        debug_textbox_state = pprint.pformat(update.debug_dict)

        debug_plot_data_variables = set().union(
            *[x.keys() for x in update.debug_plot_data],
        ) - {"t"}

        if debug_plot_data_variables:
            df = pd.DataFrame(update.debug_plot_data)
            df = df.ffill()

            fig = px.line(
                df,
                x="t",
                y=sorted(list(debug_plot_data_variables)),
            )
        else:
            fig = None

        return chatbot_state, debug_textbox_state, fig

    rtc_configuration = get_hf_turn_credentials()
    # rtc_configuration = get_cloudflare_rtc_configuration()

    stream = Stream(
        handler=UnmuteHandler(),
        modality="audio",
        mode="send-receive",
        # rtc_configuration=rtc_configuration,
        rtc_configuration=rtc_configuration,
        # additional_inputs=[gradio_chatbot],
        additional_outputs=[gradio_chatbot, gradio_debug_textbox, gradio_debug_plot],
        additional_outputs_handler=update_outputs,
        # TODO: check if clients actually get disconnected
        concurrency_limit=1,
    )

    # This variable needs to contain the Gradio UI for the autoreload to work:
    # https://www.gradio.app/guides/developing-faster-with-reload-mode
    demo = stream.ui

    # Not clear what `debug` does. It's not auto-reload.
    demo.launch(debug=False)
