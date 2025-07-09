#!/usr/bin/env python3
# /// script
# dependencies = [
#   "flask",
# ]
# ///
from flask import Flask, request, jsonify, stream_with_context, Response
import time
import json

app = Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
@stream_with_context
def chat_completions():
    # Always return the same response regardless of input
    data = request.get_json()
    msgs = data['messages']
    msgs = msgs[2:]
    print(json.dumps(data, indent=2))
    print(json.dumps(msgs, indent=2))
    
    if not msgs:
        def generate():
            yield "data: {\"choices\": [{\"delta\": {\"content\": \" .. \"}, \"index\": 0, \"finish_reason\": \"stop\"}]}\n\n"
            yield "data: [DONE]\n\n"
        return Response(generate(), mimetype='text/event-stream')

    if len(msgs) == 2:
        def generate():
            yield "data: {\"choices\": [{\"delta\": {\"content\": \"Okay, I'm searching for kouign amann. the movie film. and his director!! \"}, \"index\": 0, \"finish_reason\": null}]}\n\n"
            yield "data: [DONE]\n\n"
        return Response(generate(), mimetype='text/event-stream')


    # Fake user request: Cherche le film fantastique par le réalisateur de Kingsman
    def generate():
        #yield "data: {\"choices\": [{\"delta\": {\"content\": \" Okay, j'ai trouvé. Son réalisateur. Matthew Vaughn. Regardons ses films. !! \"}, \"index\": 0, \"finish_reason\": null}]}\n\n"
        yield "data: {\"choices\": [{\"delta\": {\"content\": \"Let's search Kingsman. the movie. in my knowledge base. !! \"}, \"index\": 0, \"finish_reason\": null}]}\n\n"
        #time.sleep(4)
        yield "data: {\"choices\": [{\"delta\": {\"content\": \" ?? Alright its director. Matthew Vaughn. Lookins at his movies. !! \"}, \"index\": 0, \"finish_reason\": null}]}\n\n"
        time.sleep(4)
        yield "data: {\"choices\": [{\"delta\": {\"content\": \" ?? Oh, of course, Stardust. Shall I play that? \"}, \"index\": 0, \"finish_reason\": \"stop\"}]}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True, host='::', port=5000)
