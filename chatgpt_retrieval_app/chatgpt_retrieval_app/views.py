from django.http import JsonResponse
from django.shortcuts import render 

import os
from . import init

chain = init()

chat_history = []
  
def get_response(prompt):
    print(f'chat history: {chat_history}')
    result = chain({"question": prompt, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((prompt, result['answer']))
    return result['answer']

def query_view(request):
    if request.method == 'POST': 
        prompt = request.POST.get('prompt') 
        response = get_response(prompt) 
        return JsonResponse({'response': response}) 

    return JsonResponse({'response': result['answer']}) 

def ui_view(request):
    return render(request, 'index.html')
