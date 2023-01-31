from typing import List
from utils.detect import YoloV7, aux
import torch 
from pydantic import BaseModel, Field
from fastapi import FastAPI, status, Response, BackgroundTasks
from wasabi import Printer
from torch import multiprocessing

basepath = "vision"
logger = Printer(timestamp=True)
model = None
threadpool = None
processes = []

app = FastAPI(
    title="INQ AI: Object Counting Service API",
    description="Receives a Video URL, processes it frame by frame, sends back True if subject has blinked",
    docs_url=f"/{basepath}/docs",
    redoc_url=f"/{basepath}/redoc",
    openapi_url=f"/{basepath}/openapi.json",
)

class CountObjects(BaseModel):
    uuid: int = Field(description="unique id to respond to", default=-1)
    url: str = Field(description="video/rtsp stream url", default="")
    timeout: int = Field(description="How Long to read from stream", default=0)
    idle: int = Field(description="Stop detection if no objects are counted in this period", default=900)
    conf_thres: float = Field(description="Confidence Threshold Value", default=0.45)
    line_begin_x: int = Field(description="Line Begin X Value", default=800)
    line_begin_y: int = Field(description="Line Begin y Value", default=1100)
    line_end_x: int = Field(description="Line End X Value", default=1800)
    line_end_y: int = Field(description="Line End y Value", default=1100)
    count: bool = Field(description="Enable/Disable counting", default=False)
    intermittent: bool = Field(description="Enable/Disable counting", default=False)
    classes: List[str] = Field(description="List of classes to act on", default=[])

# @app.on_event('startup')
def startup():
    # global model
    # model = YoloV7()
    multiprocessing.set_start_method('spawn')

@app.on_event("shutdown")
def shutdown():
    for p in processes:
        p.join()

@app.post(f"/{basepath}/object-counting", status_code=status.HTTP_201_CREATED)
async def count_objects(
    count_object: CountObjects,
    background_tasks: BackgroundTasks
) -> Response:
    print(count_object)
    try:
        with torch.no_grad():
            p = multiprocessing.Process(target=aux, kwargs={'uuid': count_object.uuid, 'source': count_object.url, 'count': count_object.count, 'intermittent': count_object.intermittent, 'classes': count_object.classes, 'nosave': False, 'timeout': count_object.timeout, 'idle': count_object.idle, 'conf_thres': count_object.conf_thres, 'line_begin': (count_object.line_begin_x, count_object.line_begin_y), 'line_end': (count_object.line_end_x, count_object.line_end_y)})
            p.start()
            processes.append(p)
        return Response(content=f"tracking")
    except Exception as ex:
        print(ex)
        return Response(content=f"failure")

class Update(BaseModel):
    uuid: int = Field(description="unique id to respond to", default=-1)
    count: int = Field(description="Line End y Value", default=0)


@app.patch(f"/{basepath}/update-server", status_code=status.HTTP_200_OK)
def update_server(update: Update):
    print(update)

if __name__ == '__main__':
    startup()
    
    count_object = CountObjects(url='rtsp://admin:Sh@y0na1@41.217.216.26:554/Streaming/Channels/401', timeout=600, idle=900, intermittent=True, count=True, classes=['sack'])
    # count_object = CountObjects(url='./testset/sack_test_1.mp4', timeout=60, idle=60, intermittent=True, count=True, classes=['sack'])
    print(model.detect(count_object.uuid, count_object.url, count=count_object.count, intermittent=count_object.intermittent, classes=count_object.classes, nosave=False, timeout=count_object.timeout, idle=count_object.idle, conf_thres=count_object.conf_thres, line_begin=(1057, 1121), line_end=(2209, 1127)))

    count_object = CountObjects(url='rtsp://admin:Sh@y0na1@41.217.216.26:554/Streaming/Channels/301', timeout=600, idle=900, intermittent=True, count=True, classes=['sack'])
    # count_object = CountObjects(url='./testset/sack_test_1.mp4', timeout=60, idle=60, intermittent=True, count=True, classes=['sack'])
    print(model.detect(count_object.uuid, count_object.url, count=count_object.count, intermittent=count_object.intermittent, classes=count_object.classes, nosave=False, timeout=count_object.timeout, idle=count_object.idle, conf_thres=count_object.conf_thres, line_begin=(1057, 1121), line_end=(2209, 1127)))
