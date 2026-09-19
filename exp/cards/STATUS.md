# 接手说明(自动队列,2026-09-18)

队列:`queue.sh`(等 B1 DONE → B2 → T1 → D1)→ `queue2.sh`(等 D1 → D2 → N2 → M1)→ `queue3.sh`(等 M1 → N1)→ `queue4.sh`(等 N1 → B1b → B2b → S1b)→ `queue5.sh`(等 S1b → D1b)→ `queue6.sh`(等 D1b → N2 重跑)→ `queue8.sh`(等第二个 end N2 → M1 → N1)→ `queue9.sh`(等 end N1 rerun2 → N2 第 3 次)→ `queue10.sh`(等 end N2 rerun3 → R1 ρ 扫描)→ `queue11.sh`(等 end R1 → N2 第 4 次)。queue7 已因误触发(匹配到旧的 end N2)与 N1 同时抢 GPU,两者 OOM;已作废。进度看 `queue.log`;每卡进度看 `exp/<card>/run.log`,最后一行 `DONE` 即完成;结果在 `exp/<card>/summary.json`(汇总)与 `results.json`(逐身份、逐 seed)。
**2026-09-19 晚:第七轮(I1/I2/I3/SW1/BG1/MB1/MB2/FO1)也已跑完并写入 LOG.md「第七轮总结」;当前最佳 = 不模糊源 + 低45°@0.15 + 中30°@0.15–0.35(0.328→0.397)。用户要求停止实验,进入整理阶段。所有队列已结束。**

**2026-09-19 状态:全部卡跑完并写入 LOG.md**(B1、B1b、B2、B2b、T1、D1、D1b、D2、N1、N2、M1、S1、S1b、R1、H1);GPU 空闲,可 Stop pod。总结见 LOG.md「第六轮总结」。
图片在容器盘 `/root/gen/<card>`(软链接),pod 重启会丢,固定 seed 可复现。
每卡完成后要做:读 summary.json → 按 next_plan 的判定规则写 LOG.md a–g(数字原样贴,不美化)→ 画图到 outputs/ → commit(用户已授权本轮自动完成)。
判定规则速查:B1 A−B 配对 p<0.05 且 ≥+0.015 → 身份经低频进入;B2 存在 θ* p<0.05 且墨镜探针降 <0.02;T1 身份迁移随 t_start 单调递减;D2 teacher_xt401 ≥ method ≈ teacher;N1 rot45 ≥ 80% 的 id_opt_K4 增益;M1 ≥4/5 prompt p<0.05 且属性不降。
若某卡崩了:看 run.log 末尾 traceback,修复后重跑(B1/T1/N2/M1 支持续跑;B2/D1/D2/N1 从头)。
