import requests

class BaiduErnie:
  def __init__(self, apikey:str, secretkey:str) -> None:
    self.apikey = apikey
    self.secretkey = secretkey
    self.access_token = None
    
  def get_access_token(self):
    url = 'https://aip.baidubce.com/oauth/2.0/token'
    params = {
      "grant_type":"client_credentials",
      "client_id": self.apikey,
      "client_secret": self.secretkey
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
      data = response.json()
      print("Request successful")
      print("Response: ", data)
      self.access_token = data['access_token']
    else:
      self.access_token = None
      print("Request failed")
      print("Response: ", response.text)
      
  def inferer(self, prompt):
    if self.access_token is None:
      self.get_access_token()
    
    url =  "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"
    params = {
      "access_token": self.access_token
    }
    body = {
      "messages": [
        {
          "role": "user",
          "content": prompt
        }
      ]
    }
    
    res = requests.post(url, params=params, json=body)
    if res.status_code == 200:
      data = res.json()
      print("Request successful")
      print("Response: ", data)
      if 'error_code' in data:
        print("Error: ", data['error_msg'])
        return None
      return data['result']
    else:
      print("Request failed")
      print("Response: ", res.text)
      return None
    
    