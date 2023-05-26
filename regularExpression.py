import re

p_summary_section = re.compile('\【[A-Z\s]+\】')
match_summary_section = p_summary_section.match('【SUMMARY OF APPLE】')
print(match_summary_section)

path_dir  = "D:/OneDrive - Obigo Inc/문서/LDA"
p_summary_section = re.compile('\【[A-Z\s]*SUMMARY[A-Z\s]*\】')
p_not_summary_section = re.compile('\【[A-Z\s]+\】')

with open(path_dir + '/US2021-0118067A1.txt', "rt", encoding="UTF-8") as f:
    file_lines = f.readlines()
    if file_lines:
        isSummarySection = False
        summaryText = ""
        for line in file_lines:
            match_summary_section = p_summary_section.match(line)
            match_not_summary_section = p_not_summary_section.match(line)

            if match_summary_section:
                print(match_summary_section)
                isSummarySection = True
            elif match_not_summary_section and isSummarySection:
                    break
            elif isSummarySection:
                # Summary구간인경우
                summaryText += line

        print("summaryText : " + summaryText)











