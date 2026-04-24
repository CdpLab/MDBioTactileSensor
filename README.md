<h1 align="center">Multimodal Perception and Three-Dimensional Force Decoupling Method for A Fingertip Bionic Tactile Sensor</h1>
<div align=center>

 <p align="center">Dapeng Chen, Peng Gao, Zhenjie Ma, Zhou Zhuang, Zhangjia Deng, Hui Zhang, Yun Ling, Xuhui Hu, Hong Zeng, Lina Wei, Jia Liu, Qiang Zhao, and Aiguo Song</p>
  <p align="center">Nanjing University of Information Science and Technology</p>
  <p align="center">Southeast University</p>
  <p align="center">Nanjing Research Institute of Simulation Technology</p>
  <p align="center">Hangzhou City University</p>

---
## <p align="center">ABSTRACT</p>
Tactile perception is crucial for robotic dexterous manipulation, human–machine interaction, and intelligent wearable systems. However, existing artificial tactile systems still face significant challenges in the coordinated acquisition of multidimensional tactile information and in the accurate decoupling of pressure and friction during contact, which severely limits the precise interpretation of tactile information in interactive processes. Inspired by the multilayer structure of human skin and the cooperative mechanism of multiple receptors, this study designed and fabricated a hierarchical tactile sensor capable of simultaneously acquiring pressure, friction, and temperature information. Experimental results demonstrate that the sensor exhibits excellent pressure and friction sensing performance, fast response and recovery times (36/49 ms), and good cyclic stability, while also enabling reliable recognition of different sliding directions and stable acquisition of contact temperature. To address the nonlinear coupling between pressure and friction during contact, an FT-Transformer-based three-dimensional force decoupling model was developed, enabling accurate mapping from multichannel sensing signals to three-dimensional contact force components with high accuracy ($R^2 \geq$  0.99) and real-time output. Furthermore, based on the dynamic temporal responses of the sensor during pressing, a hardness recognition network was constructed, achieving a high classification accuracy of 99.25\% for materials with different hardness levels. These results indicate that the proposed sensor can support higher-level tactile cognition tasks.

---
## <p align="center">Overview</p>  
Sensor model, fabrication method, control code, and model code. The following two images show the structure and prototype of our sensor.
<div align=center>
<img src="https://github.com/AILM-UX/Multidimensional-Tactile-Sensor/blob/main/figure/fig1a.jpg" alt="Image text" width="600" height="550"/>     <img src="https://github.com/AILM-UX/Multidimensional-Tactile-Sensor/blob/main/figure/fig1b.jpg" alt="Image text" width="400" height="350"/> 
</div>


---
## The Sensor folder contains the schematic diagrams of the sensors we designed 
This sensor employs a multi-layer design strategy similar to human skin. From top to bottom, it consists of a encapsulation layer, a sensing layer, and a substrate layer, enabling it to simultaneously detect pressure, friction, and temperature.

---
## Material and Methods

### Fabrication of conductive polymer force-sensing units
CB powder (Tanfeng Tech. Inc, China), MWCNTs powder (NACATE, China), and deionized water were mixed in a mass ratio of 5:2:100 and stirred thoroughly for 15 minutes to form a uniform dispersion. The PU sponge was then fully immersed in the dispersion for 1 hour to ensure that the conductive fillers formed a stable, continuous conductive network within the porous matrix. Subsequently, the impregnated PU sponge was dried in a vacuum oven at 70 °C for 3 hours and cut into 6 mm × 6 mm pieces to serve as independent force-sensing units.

### Fabrication of electrodes and temperature-sensing unit
The electrodes and temperature sensing units are integrated onto a flexible printed circuit board. Four independent and symmetrically distributed planar electrode regions are designed on the FPC, with each group corresponding to a force-sensing unit used to collect signals representing changes in the resistance of the conductive polymer during loading. A digital temperature sensor chip (BMP280, Bosch Semiconductors) is soldered with solder paste and mounted at the center of the FPC. It is connected to the signal acquisition system via a standard digital communication interface to enable synchronous temperature data acquisition.

### Fabrication of the sensor encapsulation layer and Substrate, and sensor assembly
During the preparation of the sensor’s encapsulation layer and substrate, the PDMS (Sylgard 184, Dow Corning) base material and curing agent were thoroughly mixed in a 10:1 ratio, poured into a 3D-printed PLA mold, and then vacuum-cured in a vacuum oven for 2 hours to completely remove air bubbles. The assembly was then cured at 70°C for 3 hours. After demolding, the PDMS encapsulation layer and substrate were obtained. Next, a conductive silver paste (CD02, Conduction) with a volume resistivity of 2×10⁻⁴ Ω·cm was used to bond the CB/MWCNTs/PU conductive polymer to the planar electrodes of the FPC; Finally, silicone rubber (V-705, Valigoo) was used to bond the various layers of the sensor, completing the overall assembly.

## Code
Specifically, “ADS1256” refers to the 24-bit ADC acquisition module code, “BMP280” refers to the temperature acquisition module code, “FT-CFRF” refers to the 3D force decoupling model code, and “Bi-LSTM hardness” refers to the hardness block classification model code.

## Figure
The figure section includes photos of our sensor models and prototypes, evaluation metrics for the three-force decoupling model, and prediction results for some of the test datasets.

