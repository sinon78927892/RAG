from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TemplateMessage,
    ButtonsTemplate,
    PostbackAction,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    FollowEvent,
    PostbackEvent,
    TextMessageContent
)

app = Flask(__name__)

# 設定 LINE Bot 的認證資訊
CHANNEL_ACCESS_TOKEN = 'Wz5hTGSPFsJlWWdhfkd2u9lNJ3laOWT/+HYdLgTYjSj3PBgboejmYEnyI41hF7FdJIlETGbIFi47wA5vUNCkuSGCws7UGNHT3a1lfY3RT8gxDZKM5gEQgFhi9k+UPxcyt7cnETvmluK+cFkqEPiJpQdB04t89/1O/w1cDnyilFU='
CHANNEL_SECRET = 'bfc154e4be5c1d9a4b2656addf1e479d'
# 初始化 Configuration 和 WebhookHandler
config = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers.get('X-Line-Signature')

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info(
            "Invalid signature. Please check your channel access token/channel secret."
        )
        abort(400)

    return 'OK'


@handler.add(FollowEvent)
def handle_follow(event):
    # 回應歡迎訊息
    with ApiClient(config) as api_client:
        messaging_api = MessagingApi(api_client)
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="感謝您關注我們的LINE BOT！")]
            )
        )


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    with ApiClient(config) as api_client:
        messaging_api = MessagingApi(api_client)
        
        # 如果收到特定訊息，回傳按鈕範例
        if event.message.text.lower() == 'postback':
            buttons_template = ButtonsTemplate(
                title="Postback Action",
                text="這是一個按鈕範例",
                actions=[
                    PostbackAction(
                        label="點擊這裡",
                        text="按鈕被點擊了！",
                        data="postback_action"
                    )
                ]
            )
            template_message = TemplateMessage(
                alt_text="這是一個範例按鈕樣板訊息",
                template=buttons_template
            )
            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[template_message]
                )
            )
        else:
            # 回應一般文字訊息
            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=f"你說的是: {event.message.text}")]
                )
            )


@handler.add(PostbackEvent)
def handle_postback(event):
    with ApiClient(config) as api_client:
        messaging_api = MessagingApi(api_client)
        
        if event.postback.data == "postback_action":
            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="Postback 事件被觸發！")]
                )
            )


if __name__ == "__main__":
    app.run(port=5000)