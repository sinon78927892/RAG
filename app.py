import os
from openai import OpenAI

from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage  # 載入 TextSendMessage 模組
import json

app = Flask(__name__)


@app.route("/callback", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    json_data = json.loads(body)
    print(json_data)
    try:
        line_bot_api = LineBotApi(
            'Wz5hTGSPFsJlWWdhfkd2u9lNJ3laOWT/+HYdLgTYjSj3PBgboejmYEnyI41hF7FdJIlETGbIFi47wA5vUNCkuSGCws7UGNHT3a1lfY3RT8gxDZKM5gEQgFhi9k+UPxcyt7cnETvmluK+cFkqEPiJpQdB04t89/1O/w1cDnyilFU='
        )
        handler = WebhookHandler('bfc154e4be5c1d9a4b2656addf1e479d')
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        tk = json_data['events'][0]['replyToken']  # 取得 reply token
        msg = json_data['events'][0]['message']['text']  # 取得使用者發送的訊息
        # 取出文字的前五個字元，轉換成小寫
        ai_msg = msg[:6].lower()
        reply_msg = ''
        # 取出文字的前五個字元是 hi ai:
        if ai_msg == 'hi ai:':
            # openai.api_key = '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'
            client = OpenAI(
                # This is the default and can be omitted
                api_key=os.environ.get("OPENAI_API_KEY"), )
            # 將第六個字元之後的訊息發送給 OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role":
                        "user",
                        "content":
                        msg[6:],
                    },
                ],
            )
            # 接收到回覆訊息後，移除換行符號
            reply_msg = response.choices[0].message.content
        else:
            reply_msg = msg
        text_message = TextSendMessage(text=reply_msg)
        line_bot_api.reply_message(tk, text_message)
    except Exception as e:
        print(e)
    return 'OK'


if __name__ == "__main__":
    app.run()
