# Hunting_face 使用说明

<H6>  1. 编译c++ g++ -I/usr/include/python3.6m/ c_pythhon/opencv_numpy.cpp -lpython3.6m -L/usr/include/opencv -lopencv_core -lopencv_imgcodecs -lopencv_imgproc （需要opencv 与python） ../out

<H6>  2. 测试模型 <1>. “PYTHONPATH=. ./out  test main” 使用测试数据测试需下载模型文件，与Hunting face数据。<2>. "PYTHONPATH=. ./out  test_api imgs_to_out PATH_TO_YOUR_TEST_IMG_DIR/" (需要安装python tensorflow)

代码说明：
1. Hunting_face.arch.* ： P、R、O 三个部分的网络模型
2. Hunting_face.c_python ： C++调用PYTHON的api
3. Hunting_face.generator : Hunting_face的数据生成测试或训练数据的脚本
4. Hunting_face.tools : 项目所需的小工具脚本
5. Hunting_face.test_R_to_O.py : 使用Hunting_face测试数据测试的脚本
6. Hunting_face.test_api.py : 外部图片测试用的脚本

...各个训练脚本与测试脚本
