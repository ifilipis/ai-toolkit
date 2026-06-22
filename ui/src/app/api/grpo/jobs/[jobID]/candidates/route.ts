import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();
const FLOW_GRPO_TRAINER_TYPE = 'flow_grpo_trainer';

export async function GET(request: Request, { params }: { params: { jobID: string } }) {
  const { searchParams } = new URL(request.url);
  const taskID = searchParams.get('taskID');

  try {
    const job = await prisma.job.findUnique({ where: { id: params.jobID } });
    const trainerType = job
      ? `${JSON.parse(job.job_config)?.config?.process?.[0]?.type || FLOW_GRPO_TRAINER_TYPE}`
      : FLOW_GRPO_TRAINER_TYPE;
    const candidates = await (prisma.flowGRPOCandidate as any).findMany({
      where: {
        job_id: params.jobID,
        trainer_type: trainerType,
        ...(taskID ? { vote_task_id: taskID } : {}),
      },
      orderBy: [{ created_at: 'desc' }, { order_index: 'asc' }],
    });

    return NextResponse.json({
      candidates: (candidates as any[]).map(candidate => ({
        ...candidate,
        image_url: `/api/img/${encodeURIComponent(candidate.image_path)}`,
      })),
    });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to load Flow-GRPO candidates' }, { status: 500 });
  }
}
