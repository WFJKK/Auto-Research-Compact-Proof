"""The two-page PDF summary of the 13 October runs.

    python -B results/max2-2026-10-13/build_pdf.py

Needs reportlab and the DejaVu fonts. The full write-up is README.md in this folder.
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PI

HERE = Path(__file__).resolve().parent
D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", D + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

ink, muted = colors.HexColor("#1f2328"), colors.HexColor("#57606a")
title = ParagraphStyle("t", fontName="DV-B", fontSize=14.5, leading=18, textColor=ink, spaceAfter=2)
sub = ParagraphStyle("s", fontName="DV", fontSize=8.2, leading=10.5, textColor=muted, spaceAfter=5)
h = ParagraphStyle("h", fontName="DV-B", fontSize=9.6, leading=12, textColor=ink, spaceBefore=3.5, spaceAfter=1.5)
body = ParagraphStyle("b", fontName="DV", fontSize=7.6, leading=9.7, textColor=ink, spaceAfter=2.6)
cap = ParagraphStyle("c", parent=body, fontSize=7.4, leading=9.4, textColor=muted, spaceAfter=2)
math = ParagraphStyle("m", parent=body, alignment=1, fontSize=8.8, leading=12, spaceAfter=2.5)
bullet = ParagraphStyle("bl", parent=body, leftIndent=9, bulletIndent=0, spaceAfter=1.4)
def B(t): return Paragraph(t, bullet, bulletText="•")

def table(rows, widths, left=()):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"), ("FONT", (0, 0), (-1, -1), "DV", 7.8), ("FONT", (0, 0), (-1, 0), "DV-B", 7.8),
        ("TEXTCOLOR", (0, 0), (-1, -1), ink),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, ink), ("LINEBELOW", (0, -1), (-1, -1), 0.4, muted),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4), ("TOPPADDING", (0, 0), (-1, -1), 1.4),
        *[("ALIGN", (c, 0), (c, -1), "LEFT") for c in left]]))
    return t

s = []
s.append(Paragraph("Compact proofs on max-of-2: what an agent loop finds", title))
s.append(Paragraph("Agentic loop WFJKK/Auto-Research-Compact-Proof, 13 October 2026. Two runs of Claude Fable 5.1 (effort high) through "
                   "Claude Code, identical except for the metric the agent is told to maximise. All proofs checked by the exact Python checker "
                   "(no Lean proofs yet). The full write-up, with the background and the reference proofs, is README.md in the same folder.", sub))

s.append(Paragraph("Setup", h))
s.append(B("<b>Networks.</b> max-of-2 from the proof-based tutorial: two tokens in, the larger one out, three linear maps. "
           "v64: vocabulary 64, d_model 8, 4,096 inputs. v128: vocabulary 128, d_model 16, 16,384 inputs. Ten trained networks "
           "per size: six score the rounds, four are held out for the final evaluation."))
s.append(B("<b>Proof.</b> A split tree over the inputs, each piece checked by one rule (single input, interval bound over a box of "
           "inputs, or skip), with global rules for contraction, symmetry and approximation. The checker rebuilds the network from "
           "its weights and recomputes every bound in exact rational arithmetic."))
s.append(B("<b>Length</b> L: the exact operations the checker performs, everything included (reading weights and proof file). "
           "<b>Accuracy</b>: certified inputs over all inputs. B is the length of brute force: v64 2,696,387, v128 40,407,555."))
s.append(B("<b>Agent loop.</b> Each round the agent sees the frontier, its best recipes, its last attempts with diagnostics, "
           "and returns one recipe: a program from weights to proof, run at five knob values (0 to 1) that trade length "
           "against accuracy. Ten rounds per run; recipes run in a container sandbox."))

s.append(Paragraph("Two metrics", h))
s.append(Paragraph("<b>Q</b> averages, over budgets b spaced evenly on a log scale from E (reading the weights) to B, the best accuracy a(b) certified within b:", body))
s.append(Paragraph("Q = ∫ a(b) d(ln b) / ln(B/E)", math))
s.append(Paragraph("<b>Cost of finishing</b> prices every correct input a proof leaves uncertified at the brute-force cost of one input, λ = B/n, and takes the cheapest total over all proofs:", body))
s.append(Paragraph("cost of finishing = B / min ( L + λ·U )", math))
s.append(Paragraph("with U the uncertified correct inputs. It measures complete certification only, so the best value is B over the shortest full-accuracy proof. "
                   "Q also rewards partial proofs, including where on the length axis they are placed; the hand-written cubic proof given the same "
                   "five-budget knob is the fair Q baseline.", body))

s.append(Paragraph("Results", h))
rows = [["Shortest full-accuracy proof (median)", "v64", "vs cubic", "v128", "vs cubic"],
        ["Hand-written cubic proof", "312,264", "", "2,446,083", ""],
        ["Fable 5.1 high, scored on Q", "295,698", "−5.3%", "2,112,262", "−13.6%"],
        ["      same, on held-out networks", "292,558", "−6.3%", "2,093,111", "−14.4%"],
        ["Fable 5.1 high, scored on cost of finishing", "294,854", "−5.6%", "2,117,809", "−13.4%"],
        ["      same, on held-out networks", "292,558", "−6.3%", "2,117,356", "−13.4%"]]
s.append(table(rows, left=(0,), widths=[66 * mm, 22 * mm, 20 * mm, 26 * mm, 20 * mm]))
s.append(Spacer(1, 3))
rows = [["Median Q", "v64", "v128", "Cost of finishing", "v64", "v128"],
        ["cubic proof, one length", "0.276", "0.307", "cubic proof", "8.63", "16.52"],
        ["cubic proof with a budget knob", "0.323", "0.347", "scored on Q", "9.12", "19.13"],
        ["scored on Q", "0.393", "0.404", "scored on cost of finishing", "9.14", "19.08"],
        ["scored on cost of finishing", "0.325", "0.348", "", "", ""]]
s.append(table(rows, left=(0, 3), widths=[50 * mm, 14 * mm, 14 * mm, 44 * mm, 14 * mm, 14 * mm]))
s.append(Paragraph("Full accuracy means every input the network classifies correctly (99.90 to 99.99%); every run certifies exactly that on every "
                   "network, development and held out. Medians over six development networks unless marked held out (four networks).", cap))

s.append(Paragraph("The idea the agents found", h))
s.append(Paragraph(
    "The hand-written proof contracts the three maps into one matrix M, groups inputs by the larger token b, and checks each group's "
    "smaller token a exactly, at a cost of two operations per token per output. The agents found that the margin is small only near the "
    "diagonal a ≈ b and grows with the gap b − a (measured on the weights: median margin about 3.5 next to the diagonal, about 17 far "
    "from it). So the far region is certified through coarse piecewise-constant approximations of M's rows, whose error the checker "
    "bounds exactly and which cost one merge step per segment instead of per token, and only a narrow band near the diagonal is checked "
    "exactly. Several approximations at different error levels let each label use the coarsest one its margins allow.", body))
s.append(B("Found by both runs independently: in round 3 of the run scored on Q, and in rounds 9 and 10 of the run scored on cost of "
           "finishing, after five rounds on a rotated-basis box bound worth 2%."))
s.append(B("The gain is 5 to 6% on v64 and 13 to 14% on v128 on the development networks, and 6% and 14% on the held-out ones, so the recipe is not tuned to particular weights. "
           "It grows with the vocabulary, as the share of inputs far from the diagonal does."))
s.append(B("No run went beyond it. One agent proposed a new rule (bounding far competitors through a telescoping chain of adjacent "
           "label differences); a float simulation on the weights shows its bound passes for almost no label, so it would not help."))

s.append(Paragraph("Reading the comparison", h))
s.append(B("<b>Q against cost of finishing.</b> The run scored on Q found the idea in round 3; the run scored on cost of finishing took nine. "
           "Q rewards cheap partial proofs, which forces the agent to sort inputs by difficulty, and that sorting is the idea. "
           "One run per metric, so this is suggestive, not established."))
s.append(B("<b>Q includes placement.</b> The Q run sits 0.06 to 0.07 above the knob baseline; its shorter full proof accounts for only 0.005 to 0.011 "
           "of that, the rest is better partial proofs and where they are placed. A fixed budget grid would remove the placement part."))

s.append(Paragraph(
    "Caveats: one run per metric; Python-checked only.", cap))

s.append(PageBreak())
fig = HERE / "q-and-cost-by-round.png"
iw, ih = PI.open(fig).size
w = 180 * mm
s.append(Image(str(fig), width=w, height=w * ih / iw))
s.append(Paragraph("Top: median Q per round. Bottom: cost of finishing per round, B over the shortest full-accuracy proof so far. Each round "
                   "is one Fable response; round 0 is brute force. Horizontal lines: the reference proofs (symmetry, the cubic proof, and "
                   "convexity, whose numbers are estimates), computed by reference_proofs.py.", cap))

doc = SimpleDocTemplate(str(HERE / "compact-proofs-max2-results.pdf"), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                        topMargin=9 * mm, bottomMargin=8 * mm, title="Compact proofs on max-of-2: results", author="Joshua Kames-King")
doc.build(s)
print("ok")
