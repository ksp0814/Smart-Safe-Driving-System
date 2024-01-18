# Smart Safe Driving System
<p>
    <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-14354C.svg?logo=python&logoColor=white"></a>
    <a href="#"><img alt="OnnxRuntime" src="https://img.shields.io/badge/OnnxRuntime-FF6F00.svg?logo=onnx&logoColor=white"></a>
    <a href="#"><img alt="TensorRT" src="https://img.shields.io/badge/TensorRT-49D.svg?logo=flask&logoColor=white"></a>


# ➤ 프로젝트 요약
- 전방 카메라를 통해 차선인식으로 차선이탈방지
- 전면 카메라를 이용한 운전자의 상태 파악


* **OpenCV**, **Scikit-learn**, **onnxruntime**, **pycuda** and **pytorch**.


    

<h1 id="Examples">➤ Examples</h1>

 * ***Convert Onnx to TenserRT model*** :

    Need to modify `onnx_model_path` and `trt_model_path` before converting.

    ```
    python convertOnnxToTensorRT.py -i <path-of-your-onnx-model>  -o <path-of-your-trt-model>
    ```

 * ***Quantize ONNX models*** :

    Converting a model to use float16 instead of float32 can decrease the model size.
    ```
    python onnxQuantization.py -i <path-of-your-onnx-model>
    ```

 
<h1 id="Demo">➤ Demo</h1>

* [***Demo Youtube Video***](https://www.youtube.com/watch?v=CHO0C1z5EWE)

* ***Display***

    ![!ADAS on video](https://github.com/ksp0814/lane-detection/blob/master/demo/lane-test01.jpg)

* ***Front Collision Warning System (FCWS)***

    ![!FCWS](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/demo/FCWS.jpg)

* ***Lane Departure Warning System (LDWS)***

    ![!LDWS](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/demo/LDWS.jpg)

* ***Lane Keeping Assist System (LKAS)***

    ![!LKAS](https://github.com/ksp0814/lane-detection/blob/master/demo/LKAS_01.jpg)

<h1 id="License">➤ License</h1>
WiFi Analyzer is licensed under the GNU General Public License v3.0 (GPLv3).

**GPLv3 License key requirements** :
* Disclose Source
* License and Copyright Notice
* Same License
* State Changes
