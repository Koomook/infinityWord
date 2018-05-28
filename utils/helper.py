from slackclient import SlackClient

poza_token = 'xoxp-210889208129-213610419124-214877724161-655e79305dd8ed333466196de18ded51'
sc = SlackClient(poza_token)
def slack(message, channel='#bot'):
    sc.api_call('chat.postMessage', channel = channel, text = message)