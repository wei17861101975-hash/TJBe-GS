# 🎯 TJBe-GS: Rigid Trajectory-Guided Bézier Gaussian Splatting for Dynamic 4D Reconstruction
<p align="center">
  <img src="https://img.shields.io/badge/Status-Code%20Coming%20Soon-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/3DGS-Dynamic%20Reconstruction-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/KBS-Submitted-green?style=for-the-badge"/>
</p>
---
## ✨ Demo Results
<table align="center">
  <tr>
    <td align="center"><b>Sequence 1</b></td>
    <td align="center"><b>Sequence 2</b></td>
  </tr>
  <tr>
    <td><img src="sy_result.gif" width="320" /></td>
    <td><img src="sy2_result.gif" width="320" /></td>
  </tr>
</table>
---
## 📌 About
TJBe-GS is a novel 4D reconstruction framework that explicitly decouples dynamic object motion into **prior-guided rigid displacement** and **frame-wise non-rigid deformation** using continuous Bézier curves.
### 🔑 Key Features
- 🚀 **Motion Decoupling**: Separates centroid trajectory-guided rigid translation from local non-rigid deformation
- 📈 **Continuous Deformation**: Parameterizes 3D Gaussian primitives with time-continuous Bézier curves
- 🎯 **Automated Pipeline**: PCA-enhanced robust dynamic point cloud initialization
- ⚡ **Physics Plausibility**: Motion smoothness constraint ensures inertia-compliant deformations
