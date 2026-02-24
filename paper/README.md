# IEEE Conference Paper — LeafSense

This folder contains the draft for the IEEE conference paper describing **LeafSense**, a **machine learning–based web application** for plant leaf disease detection (EfficientNet-B0 on PlantVillage).

## Files

| File | Description |
|------|-------------|
| `LeafSense_IEEE_Conference_Paper.md` | Main paper draft (Markdown). Structure follows standard IEEE sections: Abstract, Introduction, Related Work, Methodology, Implementation, Results, Conclusion, References. |

## Next Steps

1. **Fill in results**  
   After training, add your actual validation accuracy and (if you have them) precision/recall/F1 in **Section V. Results and Discussion**. You can also add a short table or figure (e.g., accuracy vs. epoch, sample predictions).

2. **Authors and affiliations**  
   Replace the placeholders at the top with author names, affiliations, and corresponding author email.

3. **References**  
   Complete the reference list and add any extra citations (e.g., PlantVillage, timm, PyTorch, your target conference’s style).

4. **Format for submission**  
   - **LaTeX:** Use the official [IEEE conference template](https://www.ieee.org/conferences/publishing/templates.html) (e.g., `IEEEtran`). Copy the content from the Markdown into the appropriate sections.  
   - **Word:** Many IEEE conferences also accept Word; use the IEEE Word template from the same page and paste/adapt the sections.

5. **Target conference**  
   Check the conference’s page limit (often 6–8 pages), font size, and reference style (usually IEEE).

## Quick Reference

- **Project repo:** [LeafSense on GitHub](https://github.com/abhishekhbihari007/leafSense)  
- **Tech summary:** See project root `TECH-STACK.md`  
- **Training:** `train.py` (PlantVillage binary); **Inference:** `app.py` (Flask + EfficientNet-B0 + plant checker + TTA)
