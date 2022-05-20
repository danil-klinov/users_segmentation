from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import UploadFileForm
from .main import run


def index(request):
    if request.method == 'POST':
        folder = 'djangoProject/files/'
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            fs = FileSystemStorage(location=folder)
            fs.save(file.name, file)
            run(folder + file.name)
            return render(request, 'index.html')
    return render(request, 'index.html')
