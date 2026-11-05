"""Shared definitions for the decision-model screen."""
import csv, json, hashlib, random

CLASSES = ['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']
FULL = {'模糊','纹理损失','过锐','过度生成'}
GRADE_NAME = {0:'无问题', 1:'P2', 2:'P1', 3:'P0'}
NAME_GRADE = {v:k for k,v in GRADE_NAME.items()}

DEFS = {
'模糊': ('图像轮廓发虚，边缘不清晰，整体看起来朦胧不清。',
        '与纹理损失的区别：模糊是整体轮廓发虚；纹理损失是局部纹理质感被涂抹，但轮廓可能仍然清楚。'),
'纹理损失': ('图像中物体的不同频率纹理、质感被涂抹，无法清晰展现。',
            '与模糊的区别：纹理损失是局部质感丢失，不一定整体发虚；模糊是整体轮廓不清。'),
'过锐': ('图像边缘出现生硬的白边、振铃或细节纹理失真，破坏自然层次与真实感。',
        '与过度生成的区别：过锐是边缘被强化出白边/振铃；过度生成是画面里凭空多出了不合理的物体或结构。'),
'花屏': ('图像出现杂乱的颜色块、线条或图像错乱的现象。',
        '花屏是大面积像素结构异常，通常整块区域内容不可辨认。'),
'过度生成': ('生成画面凭空出现多余、不合理的物体、细节或纹理，违背真实逻辑（如多余肢体、结构崩坏、漂浮物体）。',
            '与过锐的区别：过度生成是内容层面多出了不合理的东西；过锐是边缘被强化。与虚化突兀的区别：后者是虚实边界不自然。'),
'虚化突兀': ('画面虚实过渡生硬、边缘割裂，主体与背景的虚化衔接不自然。',
            '与过度生成的区别：虚化突兀是"虚实边界"问题，不一定是多出了物体。'),
}

GRADE_DEFS = ('P0：画质存在明显异常或严重损失，严重影响内容理解。\n'
              'P1：画质存在中等程度损失，部分细节丢失或瑕疵可见，但主体内容仍可辨识。\n'
              'P2：画质仅轻微损失，整体观感良好，细节丰富，可正常使用。\n'
              '无问题：不存在该缺陷。')

SYSTEM = '你是资深视频画质评测专家。严格按给定的定义和等级标准判断，只输出选项字母，不要解释。'

def options_for(defect):
    return ['P0','P1','P2','无问题'] if defect in FULL else ['P0','无问题']

def build_question(defect, opt_order):
    d, contrast = DEFS[defect]
    lines = [f'请判断这张图片在「{defect}」维度上的严重程度。',
             '', f'【定义】{defect}：{d}', f'【易混淆】{contrast}', '', '【等级标准】', GRADE_DEFS, '', '【选项】']
    letters = 'ABCD'
    for i, g in enumerate(opt_order):
        lines.append(f'{letters[i]}. {g}')
    lines += ['', f'请只输出一个字母（{"、".join(letters[:len(opt_order)])}），不要解释。']
    return '\n'.join(lines)

def load_manifest(path):
    return list(csv.DictReader(open(path)))

def make_tasks(manifest, seed=2026):
    tasks = []
    for r in manifest:
        for defect in CLASSES:
            opts = options_for(defect)
            order = opts[:]
            h = int(hashlib.md5(f"{seed}|{r['name']}|{defect}".encode()).hexdigest(), 16)
            random.Random(h).shuffle(order)
            tasks.append({
                'id': f"{r['name']}::{defect}", 'image': r['name'], 'defect': defect,
                'options': order, 'question': build_question(defect, order),
                'gold': GRADE_NAME[int(r[defect])], 'gold_grade': int(r[defect]),
                'source': r['source'], 'width': int(r['width']), 'height': int(r['height']),
            })
    return tasks
