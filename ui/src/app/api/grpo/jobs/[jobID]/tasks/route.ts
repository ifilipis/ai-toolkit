import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { getVotingInputImageMode } from '@/utils/modelCapabilities';

const prisma = new PrismaClient();
const FLOW_GRPO_TRAINER_TYPE = 'flow_grpo_trainer';

const parseIntegerField = (value: unknown, fallback: number, min: number) => {
  const raw = `${value ?? ''}`.trim();
  if (raw === '') return fallback;
  const parsed = parseInt(raw, 10);
  return Number.isFinite(parsed) ? Math.max(min, parsed) : fallback;
};

const parseFloatField = (value: unknown, fallback: number, min: number) => {
  const raw = `${value ?? ''}`.trim();
  if (raw === '') return fallback;
  const parsed = parseFloat(raw);
  return Number.isFinite(parsed) ? Math.max(min, parsed) : fallback;
};

export async function GET(request: Request, { params }: { params: { jobID: string } }) {
  const { searchParams } = new URL(request.url);
  const statusParam = searchParams.get('status') || 'requested,generating,open,voted';
  const statuses = statusParam
    .split(',')
    .map(value => value.trim())
    .filter(Boolean);

  try {
    const job = await prisma.job.findUnique({
      where: { id: params.jobID },
    });
    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }
    const jobConfig = JSON.parse(job.job_config);
    const trainerType = `${jobConfig?.config?.process?.[0]?.type || FLOW_GRPO_TRAINER_TYPE}`;
    const tasks = await (prisma.flowGRPOVoteTask as any).findMany({
      where: {
        job_id: params.jobID,
        trainer_type: trainerType,
        ...(statuses.length === 1 ? { status: statuses[0] } : { status: { in: statuses } }),
      },
      include: {
        candidates: {
          orderBy: {
            order_index: 'asc',
          },
        },
        votes: {
          orderBy: {
            created_at: 'asc',
          },
        },
      },
      orderBy: {
        created_at: 'desc',
      },
    });

    return NextResponse.json({
      tasks: (tasks as any[]).map(task => ({
        ...task,
        candidates: task.candidates.map((candidate: any) => ({
          ...candidate,
          image_url: `/api/img/${encodeURIComponent(candidate.image_path)}`,
        })),
      })),
    });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to load Flow-GRPO vote tasks' }, { status: 500 });
  }
}

export async function POST(request: Request, { params }: { params: { jobID: string } }) {
  try {
    const body = await request.json();
    const prompt = `${body.prompt || ''}`.trim();
    const negativePrompt = `${body.negative_prompt || ''}`.trim();
    const width = parseIntegerField(body.width, 1024, 64);
    const height = parseIntegerField(body.height, 1024, 64);
    const guidanceScale = parseFloatField(body.guidance_scale, 4, 0);
    const numInferenceSteps = parseIntegerField(body.num_inference_steps, 30, 1);
    const rawSampler = `${body.sampler ?? ''}`.trim();
    const rawScheduler = `${body.scheduler ?? ''}`.trim();
    const seedValue = `${body.seed ?? ''}`.trim();
    const seed = seedValue === '' ? null : parseInt(seedValue, 10);
    const ctrlImgValue = `${body.ctrl_img ?? ''}`.trim();
    const ctrlImg1Value = `${body.ctrl_img_1 ?? ''}`.trim();
    const ctrlImg2Value = `${body.ctrl_img_2 ?? ''}`.trim();
    const ctrlImg3Value = `${body.ctrl_img_3 ?? ''}`.trim();

    const job = await prisma.job.findUnique({
      where: { id: params.jobID },
    });
    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }
    const jobConfig = JSON.parse(job.job_config);
    const trainerType = `${jobConfig?.config?.process?.[0]?.type || FLOW_GRPO_TRAINER_TYPE}`;
    const requiresRolloutScheduler = trainerType === FLOW_GRPO_TRAINER_TYPE;
    const sampler =
      rawSampler ||
      (requiresRolloutScheduler
        ? 'flowmatch_step_with_logprob'
        : `${jobConfig?.config?.process?.[0]?.sample?.sampler || 'flowmatch'}`);
    const scheduler =
      rawScheduler ||
      (requiresRolloutScheduler
        ? 'flowmatch_step_with_logprob'
        : `${jobConfig?.config?.process?.[0]?.train?.noise_scheduler || ''}`);
    if (!prompt) {
      return NextResponse.json({ error: 'Prompt is required' }, { status: 400 });
    }
    if (requiresRolloutScheduler && sampler !== 'flowmatch_step_with_logprob') {
      return NextResponse.json(
        { error: `Unsupported live-voting sampler '${sampler}'. Supported values: flowmatch_step_with_logprob` },
        { status: 400 },
      );
    }
    if (requiresRolloutScheduler && scheduler !== 'flowmatch_step_with_logprob') {
      return NextResponse.json(
        { error: `Unsupported live-voting scheduler '${scheduler}'. Supported values: flowmatch_step_with_logprob` },
        { status: 400 },
      );
    }
    const modelArch = `${jobConfig?.config?.process?.[0]?.model?.arch || ''}`;
    const inputImageMode = getVotingInputImageMode(modelArch);
    let promptWithInputImage = prompt;
    if (inputImageMode === 'single') {
      if (ctrlImgValue) {
        promptWithInputImage = `${promptWithInputImage} --ctrl_img ${ctrlImgValue}`;
      }
    } else if (inputImageMode === 'multi') {
      if (ctrlImg1Value) {
        promptWithInputImage = `${promptWithInputImage} --ctrl_img_1 ${ctrlImg1Value}`;
      }
      if (ctrlImg2Value) {
        promptWithInputImage = `${promptWithInputImage} --ctrl_img_2 ${ctrlImg2Value}`;
      }
      if (ctrlImg3Value) {
        promptWithInputImage = `${promptWithInputImage} --ctrl_img_3 ${ctrlImg3Value}`;
      }
    }

    const task = await (prisma.flowGRPOVoteTask as any).create({
      data: {
        job_id: params.jobID,
        trainer_type: trainerType,
        prompt: promptWithInputImage,
        negative_prompt: negativePrompt,
        width,
        height,
        seed: Number.isNaN(seed as number) ? null : seed,
        guidance_scale: guidanceScale,
        num_inference_steps: numInferenceSteps,
        sampler,
        scheduler,
        status: 'requested',
      },
    });

    return NextResponse.json({ ok: true, task });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to create Flow-GRPO vote task' }, { status: 500 });
  }
}
