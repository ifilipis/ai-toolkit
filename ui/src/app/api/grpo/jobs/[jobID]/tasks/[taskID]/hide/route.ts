import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const ACTIVE_TASK_STATUSES = new Set(['requested', 'generating', 'open', 'voted']);
const FLOW_GRPO_TRAINER_TYPE = 'flow_grpo_trainer';

export async function POST(
  _request: Request,
  { params }: { params: { jobID: string; taskID: string } },
) {
  try {
    const job = await prisma.job.findUnique({ where: { id: params.jobID } });
    const trainerType = job
      ? `${JSON.parse(job.job_config)?.config?.process?.[0]?.type || FLOW_GRPO_TRAINER_TYPE}`
      : FLOW_GRPO_TRAINER_TYPE;
    const task = await (prisma.flowGRPOVoteTask as any).findFirst({
      where: {
        id: params.taskID,
        job_id: params.jobID,
        trainer_type: trainerType,
      },
      select: {
        id: true,
        status: true,
      },
    });

    if (!task) {
      return NextResponse.json({ error: 'Task not found' }, { status: 404 });
    }

    if (ACTIVE_TASK_STATUSES.has(task.status)) {
      return NextResponse.json({ error: 'Active tasks cannot be hidden' }, { status: 400 });
    }

    await prisma.flowGRPOVoteTask.update({
      where: {
        id: task.id,
      },
      data: {
        status: 'hidden',
      },
    });

    return NextResponse.json({ ok: true });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to hide Flow-GRPO vote task' }, { status: 500 });
  }
}
