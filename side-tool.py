from flask import Flask, request
import time
import ast
import requests
import os
import json

app = Flask(__name__)

prompt = """
You are discussing with a user. You can either anwer the user with text, or with a function call.
If you want to call a function, do a python-style call in backquotes.
Example: `turn_on()`

List of functions that you can be used:
- turn_on
- turn_off
- switch_channel("TF1") -- switch to a named tv channel
- volume_pct(63) -- set the volume level between 0 and 100%
- bank_accounts() -- get the values of bank account
"""

functions = {
    "turn_on": lambda: print("Turning on TV"),
    "turn_off": lambda: print("Turning off TV"),
    "switch_channel": lambda x: print(f"Switching to TV channel {x}"),
    "volume_pct": lambda x: print(f"Setting volume to {x}%"),
    "bank_accounts": lambda: "Bitcoin: 1 billion dollars",
}


def ast_run(node, out):
    if isinstance(node, ast.Module):
        ret = None
        for n in node.body:
            print("Running", n)
            ret = ast_run(n, out)
        return ret
    elif isinstance(node, ast.Name):
        print("Received name", node.id)
        return node.id
    elif isinstance(node, ast.Attribute):
        # this is xxx.yyy, just dumbly convert it to string assuming both operands are strings
        return f"{node.value.id}.{node.attr}"
    elif isinstance(node, ast.Call):
        func = ast_run(node.func, out)
        obj = {
            "function": func,
            "args": [ast_run(arg, out) for arg in node.args],
            "keywords": {kw.arg: ast_run(kw.value, out) for kw in node.keywords},
        }
                
        if func == "say" or func == "main.say":
            out['say'] = obj['args'][0]
            return None

        if not func in functions:
            print(f"Function {func} not found", functions.keys())
            return None
        return {"result": functions[func](*obj['args'], **obj['keywords'])}
    elif isinstance(node, ast.List):
        return [ast_run(item, out) for item in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(ast_run(item, out) for item in node.elts)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = ast_run(node.left, out)
        right = ast_run(node.right, out)
        if isinstance(node.op, ast.Add, out):
            return left + right
        elif isinstance(node.op, ast.Sub, out):
            return left - right
        elif isinstance(node.op, ast.Mult, out):
            return left * right
        elif isinstance(node.op, ast.Div, out):
            return left / right
        else:
            print("Received unknown binop", node.op)
    elif isinstance(node, ast.Expr):
        print("Received expr", node.value)
        return ast_run(node.value, out)
    else:
        print("Received unknown node", node)
        return None

def llm(msgs):
    req = {
        'model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
        'messages': msgs
    }
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['TOGETHERXYZ_APIKEY']}",
    }
    res = requests.post("https://api.together.xyz/v1/chat/completions", data = json.dumps(req), headers = headers)
    msg = res.json()['choices'][0]['message']['content']
    msg = msg.strip().rstrip()
    if msg[0] == '`' and msg[-1] == '`':
        msg = msg[1:-1]
        print('>', msg)
        tree = ast.parse(msg)
        ast_run(tree, {})
        return True

    return None


@app.route('/', methods=['POST'])
def receive_json():
    data = request.get_json()
    data = data[2:]
    if not data:
        return ""
    # First message is promopt
    # Second message is hardcoded to "Hello."
    msgs = [
            {"role": "system", "content": prompt},
    ]
    msgs += data
    print(json.dumps(msgs, indent=2))
    if llm(msgs):
        time.sleep(4)
        return "☺️"
    return ""

if __name__ == '__main__':
        app.run(host='::', port=9000, debug=True)
