# 文件名自动识别材料分类示例

## 🎯 功能说明
系统可以通过图片文件名自动识别材料分类，无需手动选择。

## 📝 命名规则
在文件名中包含材料关键词即可自动识别：

### 金属类
- `aluminum_plate.jpg` → 自动识别为铝
- `steel_gear.png` → 自动识别为钢
- `copper_pipe.gif` → 自动识别为铜
- `iron_nail.jpg` → 自动识别为铁
- `metal_surface.png` → 自动识别为钢

### 塑料类
- `plastic_case.jpg` → 自动识别为光滑塑料
- `pvc_pipe.png` → 自动识别为光滑塑料
- `abs_cover.gif` → 自动识别为光滑塑料
- `rough_plastic_surface.jpg` → 自动识别为粗糙塑料

### 玻璃类
- `glass_window.jpg` → 自动识别为透明玻璃
- `clear_bottle.png` → 自动识别为透明玻璃
- `frosted_glass_panel.gif` → 自动识别为磨砂玻璃
- `透明花瓶.jpg` → 自动识别为透明玻璃

### 木材类
- `wood_table.jpg` → 自动识别为橡木
- `oak_desk.png` → 自动识别为橡木
- `pine_chair.gif` → 自动识别为松木
- `木制家具.jpg` → 自动识别为橡木

### 织物类
- `cotton_shirt.jpg` → 自动识别为棉布
- `silk_scarf.png` → 自动识别为丝绸
- `fabric_curtain.gif` → 自动识别为棉布
- `布艺沙发.jpg` → 自动识别为棉布

### 陶瓷类
- `ceramic_vase.jpg` → 自动识别为釉面陶瓷
- `glazed_tile.png` → 自动识别为釉面陶瓷
- `rough_ceramic_pot.gif` → 自动识别为粗陶
- `陶瓷餐具.jpg` → 自动识别为釉面陶瓷

## 🔧 支持的关键词

### 中文关键词
- 铝、钢、铁、铜、金属
- 塑料、粗糙塑料
- 玻璃、透明、磨砂
- 木、橡木、松木
- 织物、棉、丝绸、布
- 陶瓷、釉面、粗陶、陶

### 英文关键词
- aluminum, steel, iron, copper, metal
- plastic, pvc, abs, poly, rough_plastic
- glass, clear, frosted_glass, frosted
- wood, oak, pine
- fabric, cotton, silk, cloth
- ceramic, glazed, rough_ceramic, pottery

## ✨ 优势
1. **无需手动选择** - 系统自动识别材料类型
2. **提高精准度** - 基于材料数据库的精准预测
3. **提升置信度** - 自动识别可提升15%-40%的置信度
4. **简化操作** - 只需在文件名中包含材料关键词

## 🎯 使用建议
1. 上传图片时，在文件名中包含材料关键词
2. 系统会自动识别并显示检测到的材料类型
3. 如果自动识别不准确，仍可手动选择材料分类
4. 手动选择的优先级高于自动识别 