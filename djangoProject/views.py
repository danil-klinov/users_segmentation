from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import csv
from shutil import make_archive
from wsgiref.util import FileWrapper
from .settings import FILES_PATH, CLUSTERING_FILES_PATH, FILE_NAME
from .forms import UploadFileForm
from .main import run


def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            fs = FileSystemStorage(location=FILES_PATH)
            fs.save(FILE_NAME, file)

            columns = []
            with open(FILES_PATH + FILE_NAME, 'r') as file:
                content = csv.reader(file)
                for row in content:
                    columns = row
                    break

            return render(request, 'index.html', {
                'columns': columns,
            })

        attributes = request.POST.getlist('attribute')
        algorithm = request.POST.get('algorithm')
        if len(attributes) > 0 and algorithm == '1':
            run(FILES_PATH + FILE_NAME, attributes)
            path_to_zip = make_archive(CLUSTERING_FILES_PATH, 'zip', CLUSTERING_FILES_PATH)
            response = HttpResponse(FileWrapper(open(path_to_zip, 'rb')), content_type='application/zip')
            response['Content-Disposition'] = 'attachment; filename="clustering.zip"'
            return response
    return render(request, 'index.html')
