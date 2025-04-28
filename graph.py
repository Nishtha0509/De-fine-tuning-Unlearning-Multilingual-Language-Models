import matplotlib.pyplot as plt
import numpy as np
import logging # 로깅 사용 예시

# Configure logging (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data from the image ---
languages = ['English', 'Korean', 'Hindi']
scores_before = [0.5429, 0.4147, 0.3913]
scores_after_eng_tuning = [0.7836, 0.6226, 0.5755]

# --- Plotting Setup ---
x = np.arange(len(languages))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, scores_before, width, label='Before Tuning', color='skyblue')
rects2 = ax.bar(x + width/2, scores_after_eng_tuning, width, label='After Tuning (Eng Only)', color='lightcoral')

# --- Add labels, title, and custom x-axis tick labels ---
ax.set_ylabel('Sentence Transformer Score (Higher is Better)', fontsize=12)
ax.set_title('Sentence Transformer Scores Before and After Tuning (English Only Data)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(languages, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, max(scores_after_eng_tuning) * 1.15)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# --- Function to add value labels on top of bars ---
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# --- Save the plot to a file --- ###### 이 부분 추가 ######
output_filename = "tuning_score_comparison.png" # 원하는 파일명 (예: .png, .jpg, .pdf)
plt.savefig(output_filename, dpi=300, bbox_inches='tight') # 파일 저장
logger.info(f"Plot saved to: {output_filename}") # 저장 완료 로그
############################################################

# --- Display the plot ---
plt.show()