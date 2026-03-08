"""
간판 이미지 중복 현황 분석 스크립트
- MD5 해시 기반 exact duplicate 탐지
- 중복 현황 리포트 출력
- 중복 그룹 CSV 저장
"""

import os
import hashlib
import json
import argparse
from pathlib import Path
from collections import defaultdict
import csv

# ── 설정 ───────────────────────────────────────────────
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


# ── 유틸 ───────────────────────────────────────────────
def md5_hash(filepath: str) -> str:
    """파일의 MD5 해시 반환"""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json_annotation(json_path: str) -> dict:
    """
    JSONL annotation 로드.
    반환: {image_path(또는 파일명): [annotation, ...]} 형태의 dict
    - 한 이미지에 ann0, ann1 등 여러 annotation이 있을 수 있으므로 리스트로 묶음
    """
    annotation_map = defaultdict(list)

    with open(json_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"  ⚠️  {line_num}번째 줄 파싱 실패, 건너뜀")
                continue

            # image_path 전체 경로 기준 + 파일명 기준 둘 다 등록
            img_path = item.get('image_path', '')
            fname = Path(img_path).name  # 'xxx.jpg'만 추출

            annotation_map[img_path].append(item)
            annotation_map[fname].append(item)   # 파일명으로도 조회 가능하게

    return dict(annotation_map)


def find_images(image_dir: str) -> list:
    """디렉토리에서 이미지 파일 경로 리스트 반환"""
    image_dir = Path(image_dir)
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(image_dir.glob(f'*{ext}'))
        paths.extend(image_dir.glob(f'*{ext.upper()}'))
    return sorted(paths)


# ── 핵심 로직 ──────────────────────────────────────────
def compute_hashes(image_paths: list, verbose: bool = True) -> dict:
    """
    각 이미지의 MD5 해시 계산
    반환: {hash: [filepath, ...]}
    """
    hash_map = defaultdict(list)
    total = len(image_paths)

    for i, path in enumerate(image_paths):
        if verbose and (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{total}] 해시 계산 중...")
        try:
            h = md5_hash(str(path))
            hash_map[h].append(str(path))
        except Exception as e:
            print(f"  ⚠️  {path.name} 읽기 실패: {e}")

    return dict(hash_map)


def analyze_duplicates(hash_map: dict, annotation_map: dict) -> dict:
    """
    중복 그룹 분석
    반환: 분석 결과 dict
    """
    total_images = sum(len(v) for v in hash_map.values())
    duplicate_groups = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    unique_hashes = {h: paths for h, paths in hash_map.items() if len(paths) == 1}

    duplicated_image_count = sum(len(v) for v in duplicate_groups.values())
    unique_image_count = len(unique_hashes)

    # 중복 그룹 내 annotation 일치 여부 확인
    anno_mismatch_groups = []
    for h, paths in duplicate_groups.items():
        filenames = [Path(p).name for p in paths]
        annos_text = [
            str(sorted([a.get('text') for a in annotation_map.get(fname, [])]))
            for fname in filenames
        ]
        if len(set(annos_text)) > 1:
            anno_mismatch_groups.append({
                'hash': h,
                'files': filenames,
                'annotations': annos_text
            })

    return {
        'total_images': total_images,
        'unique_hashes': len(unique_hashes) + len(duplicate_groups),  # 실제 고유 이미지 수
        'unique_image_count': unique_image_count,
        'duplicate_groups': len(duplicate_groups),
        'duplicated_image_count': duplicated_image_count,
        'removable_count': duplicated_image_count - len(duplicate_groups),  # 제거 가능한 수
        'anno_mismatch_count': len(anno_mismatch_groups),
        'duplicate_groups_detail': duplicate_groups,
        'anno_mismatch_groups': anno_mismatch_groups,
    }


def print_report(result: dict):
    """콘솔 리포트 출력"""
    print("\n" + "="*55)
    print("         🔍 이미지 중복 현황 분석 리포트")
    print("="*55)

    total = result['total_images']
    dup_imgs = result['duplicated_image_count']
    removable = result['removable_count']
    dup_ratio = dup_imgs / total * 100 if total > 0 else 0
    remove_ratio = removable / total * 100 if total > 0 else 0

    print(f"\n📁 전체 이미지 수          : {total:,}개")
    print(f"✅ 고유 이미지 수           : {result['unique_hashes']:,}개")
    print(f"🔁 중복에 포함된 이미지 수  : {dup_imgs:,}개  ({dup_ratio:.1f}%)")
    print(f"🗑️  제거 가능한 이미지 수   : {removable:,}개  ({remove_ratio:.1f}%)")
    print(f"\n📦 중복 그룹 수            : {result['duplicate_groups']:,}개")

    # 중복 그룹 크기 분포
    group_sizes = defaultdict(int)
    for paths in result['duplicate_groups_detail'].values():
        group_sizes[len(paths)] += 1

    if group_sizes:
        print("\n📊 중복 그룹 크기 분포:")
        for size in sorted(group_sizes.keys()):
            bar = '█' * min(group_sizes[size], 40)
            print(f"   {size}개 짜리 그룹: {group_sizes[size]:>5,}개  {bar}")

    # Annotation 불일치
    if result['anno_mismatch_count'] > 0:
        print(f"\n⚠️  같은 이미지인데 annotation이 다른 그룹: {result['anno_mismatch_count']:,}개")
        print("   → 수동 검토 필요!")
    else:
        print(f"\n✅ 중복 이미지의 annotation 모두 일치")

    print("="*55)


def save_duplicate_csv(result: dict, output_path: str):
    """중복 그룹 정보를 CSV로 저장"""
    rows = []
    for group_id, (h, paths) in enumerate(result['duplicate_groups_detail'].items(), 1):
        for rank, path in enumerate(paths):
            rows.append({
                'group_id': group_id,
                'md5_hash': h,
                'rank_in_group': rank,            # 0이 대표(keep), 나머지 제거 후보
                'action': 'keep' if rank == 0 else 'remove',
                'filepath': path,
                'filename': Path(path).name,
            })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n💾 중복 그룹 CSV 저장 완료: {output_path}")
    print(f"   (rank=0: 보존 / rank>0: 제거 후보)")


def save_clean_list(result: dict, all_paths: list, output_path: str):
    """중복 제거 후 남길 파일 목록 저장 (필터링용)"""
    remove_set = set()
    for paths in result['duplicate_groups_detail'].values():
        for p in paths[1:]:  # 첫 번째만 남기고 나머지 제거
            remove_set.add(p)

    clean_paths = [str(p) for p in all_paths if str(p) not in remove_set]

    with open(output_path, 'w', encoding='utf-8') as f:
        for p in clean_paths:
            f.write(p + '\n')

    print(f"📋 필터링 후 파일 목록 저장: {output_path}  ({len(clean_paths):,}개)")


# ── 메인 ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='간판 이미지 중복 현황 분석')
    parser.add_argument('--image_dir',  required=True, help='이미지 폴더 경로')
    parser.add_argument('--json_path',  required=True, help='annotation JSON 파일 경로')
    parser.add_argument('--output_dir', default='./duplicate_report', help='결과 저장 폴더')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 이미지 목록 수집
    print(f"\n📂 이미지 폴더: {args.image_dir}")
    image_paths = find_images(args.image_dir)
    print(f"   → {len(image_paths):,}개 이미지 발견")

    # 2. JSON annotation 로드
    print(f"\n📄 JSON 로드: {args.json_path}")
    annotation_map = load_json_annotation(args.json_path)
    print(f"   → {len(annotation_map):,}개 annotation 로드")

    # 3. 해시 계산
    print(f"\n🔐 MD5 해시 계산 중...")
    hash_map = compute_hashes(image_paths, verbose=True)

    # 4. 중복 분석
    result = analyze_duplicates(hash_map, annotation_map)

    # 5. 리포트 출력
    print_report(result)

    # 6. 결과 저장
    if result['duplicate_groups'] > 0:
        dup_csv = os.path.join(args.output_dir, 'duplicate_groups.csv')
        save_duplicate_csv(result, dup_csv)

        clean_txt = os.path.join(args.output_dir, 'clean_filelist.txt')
        save_clean_list(result, image_paths, clean_txt)

    # 7. annotation 불일치 그룹 저장
    if result['anno_mismatch_count'] > 0:
        mismatch_path = os.path.join(args.output_dir, 'annotation_mismatch.json')
        with open(mismatch_path, 'w', encoding='utf-8') as f:
            json.dump(result['anno_mismatch_groups'], f, ensure_ascii=False, indent=2)
        print(f"⚠️  annotation 불일치 그룹 저장: {mismatch_path}")

    print("\n✅ 분석 완료!\n")


if __name__ == '__main__':
    main()