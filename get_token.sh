curl --location --request POST 'https://www.linkedin.com/oauth/v2/accessToken' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--data-urlencode 'grant_type=authorization_code' \
--data-urlencode 'code={authorization_code_from_step2_response}' \
--data-urlencode 'client_id={77uh8g2yx5z7y7}' \
--data-urlencode 'client_secret={WPL_AP1.DDcai5z5ZSr81c5p.eZvDmg==}' \
--data-urlencode 'redirect_uri={your_callback_url}'
