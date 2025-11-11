### Need to download the model file and put it in a File or model folder and copy the path to code
## and the model file download path is: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/tree/main


## TEST IN POSTMAN
Field,Value
Method,POST
URL,http://localhost:5007/transcribe
Body,form-data
Key,file → File
Value,Upload sample.wav

FieldValueMethodPOSTURLhttp://localhost:5007/transcribeBodyform-dataKeyfile → FileValueUpload sample.wav

EXPECTED JSON RESPONSE
json{
  "transcription": "Artificial intelligence is revolutionizing a world with machines that learn and tackle complex challenges seamlessly."
}

HEALTH CHECK
textGET http://localhost:5007/health
→ {"status": "healthy"}
