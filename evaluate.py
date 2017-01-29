import argparse
import csv
import os
import shutil
import subprocess as sp
import sys
import time
import datetime
import pickle
from itertools import combinations
from collections import Counter

INTERPRETER = '/cs/wetlab/Alon/hackathon/slitherenv/bin/python3.5'

if not sys.executable == INTERPRETER:  # divert to the "right" interpreter
    scriptpath = os.path.abspath(sys.modules[__name__].__file__)
    sp.Popen([INTERPRETER, scriptpath] + sys.argv[1:]).wait()
    exit()


code_path = os.path.abspath(os.path.expanduser('~/Dropbox/workspace/APML/hackathon')) + '/'
archive_path = os.path.abspath(os.path.expanduser('~/Dropbox/workspace/APML/hackathon/archive')) + '/'


def parse_policies(policies_file):
    return {p['id'] : (p['short_name'], p['paragraph'])
            for p in csv.DictReader(open(policies_file))}


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('policies_file', type=str, help="file name for a csv file with <id>,<short_name>,<paragraph>")
    p.add_argument('scoreboard_files', type=str, help="comma-sparated file names for the scoreboard to be written to")
    p.add_argument('--score_archive', '-a', default=None, type=str,
                   help="file name to which results are archived after every stage")
    p.add_argument('--base_duration', '-d', type=int, help="number of rounds in basic game", default=1e6)
    p.add_argument('--handle_previous_run', '-r', choices=['rm','arch'], default='rm',
                   help="whether previous results should be removed or archived")
    args = p.parse_args()
    folders = [code_path + f for f in ['output','scripts','states']]
    if args.handle_previous_run == 'rm':
        for f in folders:
            try: shutil.rmtree(f)
            except IOError: pass
    else:
        prev = [int(x) for x in os.listdir(archive_path) if os.path.isdir(archive_path+x)]
        mvto = archive_path + str(1 + max(prev)) if prev else '0'
        for f in folders:
            try: shutil.move(f, mvto)
            except IOError: pass

    for f in folders: os.mkdir(f)
    return args.policies_file, args.scoreboard_files.split(','), args.score_archive, args.base_duration


def script_string(python_comm, output_file, ncpu=3, mem='1G'):
    return  """#! /bin/bash
#SBATCH --cpus-per-task={ncpu}
#SBATCH --output={ofile}
#SBATCH --mem-per-cpu={mem}

{pcom}""".format(ncpu=ncpu, ofile=output_file,mem=mem, pcom=python_comm)


def test_single(policies, script_folder=None, repeats=3, dur=int(1e6)):
    if script_folder is None:
        script_folder = code_path + '/scripts/'
    result_files = []
    for r in range(repeats):
        for pid in policies:
            script_name = script_folder + pid + '.%i.single.bash' % r
            res_file = '%s/output/single_%s_%i.res' % (code_path, pid, r)
            out_file = '%s/output/single_%s_%i.out' % (code_path, pid, r)
            result_files.append(res_file)
            pcom = ('python3 {p}/slither.py -P "{pid}()" -D {d} -r '
                    '-sl -o {o} -m 1').format(p=code_path,pid=pid,d=str(dur),o=res_file)
            with open(script_name, 'w') as script_file:
                script_file.write(script_string(pcom,out_file))
            out = sp.Popen(['sbatch', script_name], stdout=sp.PIPE).communicate()
            jid = out[0].decode('utf').split(' ')[-1].strip()
            print("submitted job %s (%s) for single test (%i) of policy %s" % (jid, script_name, r, pid))
    return result_files


def test_one_player(policies, best_state_files, script_folder=None, repeats=3, dur=int(2e6)):
    if script_folder is None:
        script_folder = code_path + '/scripts/'
    result_files = []
    for r in range(repeats):
        for pid in policies:
            script_name = script_folder + pid + '.%i.one-player.bash' % r
            res_file = '%s/output/one-player_%s_%i.res' % (code_path, pid, r)
            out_file = '%s/output/one-player_%s_%i.out' % (code_path, pid, r)
            result_files.append(res_file)
            pcom = ('python3 {p}/slither.py -P "{pid}(load_from={sp})" -D {d} -r '
                    '-sl -o {o} -m 2').format(p=code_path,pid=pid,sp=best_state_files[pid],d=str(dur*2),o=res_file)
            with open(script_name, 'w') as script_file:
                script_file.write(script_string(pcom, out_file))
            out = sp.Popen(['sbatch', script_name], stdout=sp.PIPE).communicate()
            jid = out[0].decode('utf').split(' ')[-1].strip()
            print("submitted job %s (%s) for one-player test (%i) of policy %s" % (jid, script_name, r, pid))
    return result_files


def test_multiplayer(policies, best_state_files, script_folder=None, k=3, dur=int(3e6)):
    if script_folder is None:
        script_folder = code_path + '/scripts/'
    result_files = []
    for i, subp in enumerate(combinations(policies, k)):
        script_name = script_folder + '%i.multi-player.bash' % i
        res_file = '%s/output/multi-player_%i.res' % (code_path, i)
        out_file = '%s/output/multi-player_%i.out' % (code_path, i)
        result_files.append(res_file)
        pstring = ';'.join('%s(load_from=%s)' % (p, best_state_files[p]) for p in subp)
        pcom = ('python3 {p}/slither.py -P "{ps}" -D {d} -r '
                '-sl -o {o} -m 0').format(p=code_path, ps=pstring, d=str(dur*3), o=res_file)
        with open(script_name, 'w') as script_file:
            script_file.write(script_string(pcom, out_file))
        out = sp.Popen(['sbatch', script_name], stdout=sp.PIPE).communicate()
        jid = out[0].decode('utf').split(' ')[-1].strip()
        print("submitted job %s (%s) for multi-player test %i of policies %s" % (jid, script_name, i, ','.join(subp)))
    return result_files


def merge_results(result_files, type=''):
    res_files = [r for r in result_files] # just a copy
    all = []
    while res_files:
        r = res_files.pop()
        try:
            with open(r) as R:
                for res in csv.DictReader(R):
                    if res['policy'] == 'AvoidCollisions': continue
                    res['type'] = type
                    all.append(res)
        except IOError:
            res_files.append(r)
            time.sleep(1)
    return all


def get_best_state_file(results):
    best_map = {}
    for r in results:
        if r['policy'] not in best_map: best_map[r['policy']] = ('', -1)
        if best_map[r['policy']][1] < float(r['score']):
            best_map[r['policy']] = (r['state_file_path'], float(r['score']))
    return {k : v[0] for k,v in best_map.items()}


def update_scoreboard(results, sb_files, archive_file):
    if archive_file:
        with open(archive_file, 'wb') as A: pickle.dump(results, A)
    fns, reformatted, tcnt = set([]), {}, Counter()
    for r in results:
        tcnt[(r['policy'], r['type'])] += 1
        if r['policy'] not in reformatted: reformatted[r['policy']] = {}
        fn = '%s%i' % (r['type'], tcnt[(r['policy'], r['type'])])
        reformatted[r['policy']][fn] = float(r['score'])
        fns.add(fn)
    fns = sorted(fns)
    for p, scores in reformatted.items():
        scores['average'] = sum(scores.values()) / len(scores)
    fns = ['policy'] + fns + ['average']
    policy_order = sorted(reformatted, key=lambda x:reformatted[x]['average'], reverse=True)
    for sb_file in sb_files:
        with open(sb_file, 'w') as SB:
            wrtr = csv.DictWriter(SB, fieldnames=fns)
            wrtr.writeheader()
            for p in policy_order:
                scores = reformatted[p]
                scores['policy'] = p
                wrtr.writerow(scores)


if __name__ == '__main__':
    print('start time: %s' % str(datetime.datetime.now()))
    pfile, sb_files, afile, dur = parse_input()
    policies = parse_policies(pfile)
    results = merge_results(test_single(policies, repeats=5, dur=dur), 'single')
    print('stage 1 time: %s' % str(datetime.datetime.now()))
    update_scoreboard(results, sb_files, afile)
    best_state_files = get_best_state_file(results)
    results.extend(merge_results(test_one_player(policies, best_state_files, repeats=5,dur=dur*2), 'one-player'))
    print('stage 2 time: %s' % str(datetime.datetime.now()))
    update_scoreboard(results, sb_files, afile)
    best_state_files = get_best_state_file(results)
    results.extend(merge_results(test_multiplayer(policies, best_state_files, k=3,dur=dur*3), 'multi-player'))
    print('stage 3 time: %s' % str(datetime.datetime.now()))
    update_scoreboard(results, sb_files, afile)

