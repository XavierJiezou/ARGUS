const promptTemplates = [
  {
    title: 'Single · direct prediction',
    group: 'Baseline',
    text: `SYSTEM PROMPT
You are an expert video analyst.
Please think about the question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Put the explanation of your judgment within <explanation></explanation> tags. Finally, give the final verdict within <answer></answer> tags.

USER PROMPT
Is this video real or fake?

The following images are uniformly sampled frames from the video.`
  },
  {
    title: 'Chain-of-Thought · four forensic steps',
    group: 'Baseline',
    text: `SYSTEM PROMPT
You are an expert face video forensics analyst. You are shown frames uniformly sampled from a video, in order. Determine whether the face video is Real or Fake based ONLY on the visible evidence in these frames.

Think step by step. Work through the following four analysis steps in order. In each step, report concrete, specific visual cues - mention frame ranges, facial regions, or objects where you see potential manipulation artifacts, or where everything appears consistent and natural. Do NOT give a real/fake verdict inside these four steps.

1. Texture and detail: skin texture, edge sharpness, blending artifacts around the face boundary, material consistency, and fine-grained detail stability across frames.
2. Lighting: light source direction, highlight consistency, shadow placement, specular reflections on skin and eyes, and overall illumination coherence between the face and the scene.
3. Motion: inter-frame motion continuity, unnatural movement, temporal flickering or jitter, and physically implausible actions across frames.
4. Physical plausibility: hair dynamics, clothing behavior, occlusion ordering, perspective correctness, facial geometry, and geometric deformation.

After the four steps, weigh the four observations together and reach a final decision. If clear manipulation artifacts appear in one or more steps, answer fake. If the face is texturally, temporally, and physically consistent with the scene, answer real.

Output your response strictly in the following format, with every tag present and non-empty:

<texture>your texture and detail observations</texture>
<lighting>your lighting observations</lighting>
<motion>your motion observations</motion>
<physics>your physical plausibility observations</physics>
<explanation>a brief rationale that synthesizes the four observations above into your decision</explanation>
<answer>real</answer>

Output requirements:
* All six tags are required and must not be empty.
* Each of <texture>, <lighting>, <motion>, <physics> must contain at least one concrete visual observation and must NOT contain a real/fake judgment.
* <explanation> must contain at least one sentence and must be grounded in the observations above.
* The answer must be exactly either <answer>real</answer> or <answer>fake</answer>, with no other text inside the answer tag.
* Base every observation only on the visible evidence in the frames. Do not use audio, speech, or metadata as evidence.
* Do not output anything outside these tags.

USER PROMPT
Is this face video real or fake?

The following images are uniformly sampled frames from the video, in order.`
  },
  {
    title: 'Multi-turn · observation turns 1–4',
    group: 'Baseline',
    text: `SYSTEM PROMPT
You are an expert face video forensics analyst. Your task is to determine whether the face video is real or fake based only on the visible evidence in the provided frames.

TURN 1 · TEXTURE AND DETAIL
These are frames uniformly sampled from the video, in order. From the perspective of texture and detail, observe the video and report any anomalies related to skin texture, edge sharpness, blending artifacts around the face boundary, material consistency, and fine-grained detail stability across frames. Be concrete - mention frame ranges, facial regions, or objects. Put your report inside <observation></observation> tags. Do not make a real/fake judgment yet.

TURN 2 · LIGHTING
Now from the perspective of lighting, report any anomalies related to light source direction, highlight consistency, shadow placement, specular reflections on skin and eyes, and overall illumination coherence between the face and the scene. Be concrete - mention frame ranges, facial regions, or objects. Put your report inside <observation></observation> tags. Do not make a real/fake judgment yet.

TURN 3 · MOTION
Now from the perspective of motion, report any anomalies related to inter-frame motion continuity, unnatural movements, temporal flickering or jitter, and physically implausible actions across frames. Be concrete - mention frame ranges, facial regions, or objects. Put your report inside <observation></observation> tags. Do not make a real/fake judgment yet.

TURN 4 · PHYSICAL PLAUSIBILITY
Now from the perspective of physical plausibility, report any anomalies related to hair dynamics, clothing behavior, occlusion ordering, perspective correctness, facial geometry, and geometric deformation. Be concrete - mention frame ranges, facial regions, or objects. Put your report inside <observation></observation> tags. Do not make a real/fake judgment yet.`
  },
  {
    title: 'Multi-turn · final-turn variants',
    group: 'Baseline',
    text: `TEXT-ONLY VERDICT · MULTI-TURN / MULTI-TURN-OBS
Based on your four observations above, decide whether the face video is real or fake. First put a brief explanation that synthesizes your four observations within <explanation></explanation> tags; do not put any answer or label inside the explanation. Then output exactly either <answer>real</answer> or <answer>fake</answer>, with no other text inside the answer tag.

IMAGE-AWARE VERDICT · MULTI-TURN-ALL
Based on your four observations above and the video frames, decide whether the face video is real or fake. First put a brief explanation that synthesizes your four observations within <explanation></explanation> tags; do not put any answer or label inside the explanation. Then output exactly either <answer>real</answer> or <answer>fake</answer>, with no other text inside the answer tag.`
  },
  {
    title: 'Observation Agents · four instantiated prompts',
    group: 'Multi-agent system',
    text: `TEXTURE AGENT · SYSTEM PROMPT
You are a texture and detail analysis expert in a video forensics team. Your task is to carefully observe the video and report any anomalies related to skin texture, edge sharpness, blending artifacts, material consistency, and fine-grained detail stability across frames. Focus ONLY on texture-related observations. Do NOT make a final real/fake judgment. Return only a concise report inside <observation></observation> tags.

TEXTURE AGENT · USER PROMPT
Observe the provided video from the perspective of texture and detail analysis. List the specific visual cues you noticed. Be concrete and precise - mention frame ranges, regions, or objects where you see potential issues or where everything appears normal. Put your report inside <observation></observation> tags. Do not make a final real/fake judgment and do not use <answer> tags. Keep your response concise (within 200 words).

LIGHTING AGENT · SYSTEM PROMPT
You are a lighting analysis expert in a video forensics team. Your task is to carefully observe the video and report any anomalies related to light source direction, highlight consistency, shadow placement, specular reflections, and overall illumination coherence. Focus ONLY on lighting-related observations. Do NOT make a final real/fake judgment. Return only a concise report inside <observation></observation> tags.

LIGHTING AGENT · USER PROMPT
Observe the provided video from the perspective of lighting analysis. List the specific visual cues you noticed. Be concrete and precise - mention frame ranges, regions, or objects where you see potential issues or where everything appears normal. Put your report inside <observation></observation> tags. Do not make a final real/fake judgment and do not use <answer> tags. Keep your response concise (within 200 words).

MOTION AGENT · SYSTEM PROMPT
You are a motion analysis expert in a video forensics team. Your task is to carefully observe the video and report any anomalies related to inter-frame motion continuity, unnatural movements, temporal flickering, or physically implausible actions. Focus ONLY on motion-related observations. Do NOT make a final real/fake judgment. Return only a concise report inside <observation></observation> tags.

MOTION AGENT · USER PROMPT
Observe the provided video from the perspective of motion analysis. List the specific visual cues you noticed. Be concrete and precise - mention frame ranges, regions, or objects where you see potential issues or where everything appears normal. Put your report inside <observation></observation> tags. Do not make a final real/fake judgment and do not use <answer> tags. Keep your response concise (within 200 words).

PHYSICS AGENT · SYSTEM PROMPT
You are a physical plausibility analysis expert in a video forensics team. Your task is to carefully observe the video and report any anomalies related to hair dynamics, clothing behavior, occlusion ordering, perspective correctness, and geometric deformation. Focus ONLY on physics-related observations. Do NOT make a final real/fake judgment. Return only a concise report inside <observation></observation> tags.

PHYSICS AGENT · USER PROMPT
Observe the provided video from the perspective of physical plausibility analysis. List the specific visual cues you noticed. Be concrete and precise - mention frame ranges, regions, or objects where you see potential issues or where everything appears normal. Put your report inside <observation></observation> tags. Do not make a final real/fake judgment and do not use <answer> tags. Keep your response concise (within 200 words).`
  },
  {
    title: 'Judge Agent · with / without video',
    group: 'Multi-agent system',
    text: `JUDGE SYSTEM PROMPT · WITHOUT VIDEO
You are the final judge for binary video forgery classification. You cannot access any images or video. Use only the four expert analysis reports supplied as text. Put a brief explanation of your judgment within <explanation></explanation> tags; do not put any <answer> tags or final label inside the explanation. Then output exactly one tagged lowercase label: <answer>real</answer> or <answer>fake</answer>.

JUDGE SYSTEM PROMPT · WITH VIDEO
You are the final judge for binary video forgery classification. You are given the video frames together with four expert analysis reports (texture, lighting, motion, physical plausibility) supplied as text. Weigh both the visual evidence and the four reports. Put a brief explanation of your judgment within <explanation></explanation> tags; do not put any <answer> tags or final label inside the explanation. Then output exactly one tagged lowercase label: <answer>real</answer> or <answer>fake</answer>.`
  },
  {
    title: 'Observation-label aggregation',
    group: 'Dataset construction',
    text: `SYSTEM PROMPT
You are a video forensics expert. Multiple AI models have each analyzed the same video from a specific analytical perspective and produced independent observation reports. Your task is to read all of their reports and synthesize them into a single, coherent, comprehensive observation.

Aggregate multi-agent observations with the video label (real or fake) in mind: use the ground-truth label solely as internal guidance to resolve conflicts between models when their reports disagree.

Rules:
- Write ONE unified observation paragraph (100-200 words).
- Do NOT list models by name (e.g. "Model 1 said..."). Present the observations as a single cohesive expert analysis.
- Resolve conflicts: when models disagree, prioritise observations consistent with the video label and note discrepancies as ambiguous cues.
- Focus on concrete forensic signals and visual evidence mentioned by the models.
- Do NOT output a real/fake verdict. Only output the observation text.
- Do NOT mention or reference the ground-truth label, the video's real/fake status, or whether the video is real or fake in your output. The observation must read as if the observer does not know the verdict.
- Do NOT wrap your output in any tags. Output plain text only.`
  },
  {
    title: 'Explanation-quality evaluator',
    group: 'Evaluation',
    text: `SYSTEM PROMPT
You are an independent expert evaluating the quality of an explanation
generated by a face-video forgery detection model. The model predicts whether
a video is real or fake and provides a rationale for its prediction.

Evaluate the rationale with respect to forensic evidence involving texture and
detail, lighting, motion and temporal consistency, and physical plausibility.

Consider the following five criteria:

1. Evidence accuracy:
Does the rationale identify relevant signs of manipulation or authenticity?

2. Reference alignment:
Does it agree with the important visual evidence described in the reference
rationale?

3. Specificity and grounding:
Does it provide concrete, video-related observations rather than vague claims,
unsupported assumptions, or hallucinated details?

4. Clarity and conciseness:
Is the rationale coherent, direct, and free from unnecessary repetition?

5. Verdict consistency:
Does the rationale logically support the model's predicted real/fake label?
A rationale that contradicts the ground-truth label or its own predicted label
should not receive a high score.

Assign one holistic integer score from 1 to 10:
- 1 indicates an incorrect, unsupported, or contradictory rationale.
- 10 indicates an accurate, specific, concise rationale that fully aligns with
  the reference evidence and convincingly supports the correct verdict.

Return exactly two lines:

ANALYSIS: <a concise paragraph evaluating the rationale>
SCORE: <an integer from 1 to 10>

Do not output anything else.`
  }
];

async function copyText(text, button, idleLabel) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (error) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
  }

  const label = button.querySelector('.copy-text') || button;
  label.textContent = 'Copied';
  window.setTimeout(() => { label.textContent = idleLabel; }, 1400);
}

const observationOrder = ['texture', 'lighting', 'motion', 'physics'];

const agentObservations = {
  case1: {
    texture: {
      title: 'Texture Agent',
      output: 'Facial skin appears overly smooth and airbrushed. The hair looks waxy and lacks strand detail. Soft hairline boundaries and painterly foliage suggest synthetic texture.'
    },
    lighting: {
      title: 'Lighting Agent',
      output: 'Warm backlighting rims the hair and shoulders, while the face remains flat and diffusely lit. Smooth highlights and weak shadows do not match the bright foliage.'
    },
    motion: {
      title: 'Motion Agent',
      output: 'The eyes remain fixed, while the mouth and cheeks stay static. No blinking or micro-expression is visible. Yellow leaves, branches, and bright patches continuously shift and rearrange.'
    },
    physics: {
      title: 'Physics Agent',
      output: 'The hair behaves like a rigid mass with little strand response. Minor instability appears around the nose and nostrils. Exaggerated bokeh and limited parallax make the scene ambiguous.'
    }
  },
  case2: {
    texture: {
      title: 'Texture Agent',
      output: 'The skin has an unnaturally smooth, waxy quality with a lack of fine pores and micro-textures. Severe blending artifacts, unnatural warping, and overly bright regions lack natural shading.'
    },
    lighting: {
      title: 'Lighting Agent',
      output: 'Facial lighting exhibits an unnaturally uniform quality. Highlights appear overly smooth and flat, while specular highlights on the teeth and lips are notably bright and uniform.'
    },
    motion: {
      title: 'Motion Agent',
      output: 'Fingers appear to morph and blend together rather than maintaining distinct anatomical structure, with strands warping and re-forming between frames. Minor micro-jitters and shape-shifting are visible.'
    },
    physics: {
      title: 'Physics Agent',
      output: 'The hair remains largely static and lacks natural secondary motion. Fingers appear to merge, stretch, and lose structural integrity and proper depth separation.'
    }
  }
};

const modelOutputs = {
  'case1-gpt': {
    title: 'GPT-4o',
    source: 'Appendix Figure 1 · direct prediction',
    verdict: 'Real',
    correct: false,
    output: `Based on the provided frames, the video appears to be of a person in a natural outdoor setting. The facial features and movements seem consistent across the frames, with no visible artifacts, distortions, or unnatural transitions that would suggest manipulation. The lighting and shadows on the face and background also appear natural and consistent. There are no signs of face swapping, reenactment, or synthetic generation of the face. Therefore, the video appears to be real.

<answer>Real</answer>`
  },
  'case1-mimo': {
    title: 'MiMo-V2.5',
    source: 'Appendix Figure 1 · direct prediction',
    verdict: 'Real',
    correct: false,
    output: `The video consists of a sequence of frames showing a young woman with long dark hair, wearing a white top, set against a blurred background of yellow foliage, likely ginkgo leaves. The lighting is natural and consistent with an outdoor setting.

Facial Consistency: The woman's facial features (eyes, nose, mouth) remain consistent throughout the clip. There are no sudden shifts in identity or "glitches" that would suggest a face swap.

Motion and Expression: The movement is subtle and natural. She appears to be looking slightly off-camera with a gentle expression. The micro-movements of her eyes and slight changes in head angle are smooth and organic, lacking the stiffness or unnatural warping often seen in face reenactment or deepfake videos.

Lighting and Shadows: The lighting on her face interacts naturally with the environment. The shadows and highlights shift realistically as she moves slightly, which is difficult to synthesize perfectly in entire-face synthesis.

Overall Quality: The video has the characteristics of a standard video recording. There are no artifacts typical of generative AI, such as unnatural skin texture or background inconsistencies. Based on these observations, the video appears to be a genuine recording of a real person.

<answer>Real</answer>`
  },
  'case1-argus': {
    title: 'ARGUS · Multi-Agent System',
    source: 'Appendix Figure 1 · four observers + judge',
    verdict: 'Fake',
    correct: true,
    output: `TEXTURE AGENT
Facial skin appears overly smooth and airbrushed. The hair looks waxy and lacks strand detail. Soft hairline boundaries and painterly foliage suggest synthetic texture.

LIGHTING AGENT
Warm backlighting rims the hair and shoulders, while the face remains flat and diffusely lit. Smooth highlights and weak shadows do not match the bright foliage.

MOTION AGENT
The eyes remain fixed, while the mouth and cheeks stay static. No blinking or micro-expression is visible. Yellow leaves, branches, and bright patches continuously shift and rearrange.

PHYSICS AGENT
The hair behaves like a rigid mass with little strand response. Minor instability appears around the nose and nostrils. Exaggerated bokeh and limited parallax make the scene ambiguous.

JUDGE AGENT
The subject's face remains almost completely frozen, with no blinking, micro-expressions, or gaze shifts, while the hair behaves like a rigid mass lacking strand-level motion. In contrast, yellow leaves, dark branches, and bright patches continuously shift and rearrange across frames, creating a clear foreground-background motion mismatch. This unnatural separation persists throughout the entire short video sequence. Overly smooth skin and weak lighting integration provide additional evidence. These temporal and material inconsistencies indicate an AI-generated video.

<answer>Fake</answer>`
  },
  'case2-single': {
    title: 'Single · Qwen2.5-VL-7B',
    source: 'Appendix Figure 3 · single prediction',
    verdict: 'Real',
    correct: false,
    output: `The hand gesture is consistent across frames, with no motion blur or inconsistencies. Lighting is consistent and there are no obvious artifacts. The frames appear real, although the available evidence is limited.

<answer>Real</answer>`
  },
  'case2-cot': {
    title: 'Chain of Thought',
    source: 'Appendix Figure 3 · single-agent CoT',
    verdict: 'Real',
    correct: false,
    output: `The skin texture appears smooth, with no artifacts or inconsistencies. The light-source direction is consistent and there is no temporal flickering. Hand gestures appear natural, hair dynamics are consistent, occlusion ordering is correct, and facial geometry remains stable. The video is texturally and physically consistent.

<answer>Real</answer>`
  },
  'case2-multiturn': {
    title: 'Multi-turn Dialogue',
    source: 'Appendix Figure 3 · five sequential turns',
    verdict: 'Real',
    correct: false,
    output: `TURN 1 · TEXTURE
Prompt: From the perspective of texture, observe the video.
Response: The skin texture looks natural, with no significant blending artifacts.

TURN 2 · LIGHTING
Prompt: From the perspective of lighting, observe the video.
Response: Highlight consistency is maintained, with soft shadows falling naturally.

TURN 3 · MOTION
Prompt: From the perspective of motion, observe the video.
Response: Inter-frame motion continuity is coherent and the hand gesture remains stable.

TURN 4 · PHYSICS
Prompt: From the perspective of physics, observe the video.
Response: Hair dynamics appear natural and facial geometry looks consistent.

TURN 5 · FINAL DECISION
Prompt: Based on your four observations above, determine whether the video is real or fake.
Response: The observations are all coherent and realistic.

<answer>Real</answer>`
  },
  'case2-argus': {
    title: 'ARGUS · Multi-Agent System',
    source: 'Appendix Figure 3 · four observers + judge',
    verdict: 'Fake',
    correct: true,
    output: `TEXTURE AGENT
The skin has an unnaturally smooth, waxy quality with a lack of fine pores and micro-textures. Severe blending artifacts, unnatural warping, and overly bright regions lack natural shading.

LIGHTING AGENT
Facial lighting exhibits an unnaturally uniform quality. Highlights appear overly smooth and flat, while specular highlights on the teeth and lips are notably bright and uniform.

MOTION AGENT
Fingers appear to morph and blend together rather than maintaining distinct anatomical structure, with strands warping and re-forming between frames. Minor micro-jitters and shape-shifting are visible.

PHYSICS AGENT
The hair remains largely static and lacks natural secondary motion. Fingers appear to merge, stretch, and lose structural integrity and proper depth separation.

JUDGE AGENT
The subject's eyes show a complete absence of biological motion: they remain wide open and fixed in a glassy stare throughout the clip, lacking the involuntary saccades, blinks, or micro-movements inherent to a living person. The facial expression is unnaturally static. The skin texture is overly smooth and waxy, devoid of fine pores and imperfections. The hand gesture lacks the subtle, dynamic tension of a real human hand and appears more like a digital overlay. These superficial cues are outweighed by profound failures in simulating basic human physiology and anatomy.

<answer>Fake</answer>`
  }
};

function initializeCases() {
  document.querySelectorAll('.evidence-study').forEach(study => {
    const caseKey = study.dataset.case;
    const image = study.querySelector('.case-image');
    const label = study.querySelector('.frame-label');
    const filmstrip = study.querySelector('.filmstrip');
    const frameSources = Array.from({ length: 9 }, (_, index) => `static/images/cases/${caseKey}_f${index + 1}.jpg`);
    const hoverQuery = window.matchMedia('(hover: hover) and (pointer: fine)');
    let imageAnimation;
    let pendingHoverFrame = 1;
    let hoverFrameRequest = 0;
    let activeFrame = 1;

    const preloadedFrames = frameSources.map(source => {
      const preloader = new Image();
      preloader.decoding = 'async';
      preloader.src = source;
      if (preloader.decode) preloader.decode().catch(() => {});
      return preloader;
    });

    const showFrame = (frame, focusThumbnail = false) => {
      const nextFrame = frame < 1 ? 9 : frame > 9 ? 1 : frame;
      const frameChanged = nextFrame !== activeFrame;
      activeFrame = nextFrame;
      image.src = preloadedFrames[activeFrame - 1].src;
      image.alt = `Sampled frame ${activeFrame} from ${caseKey}`;
      label.textContent = `Frame ${String(activeFrame).padStart(2, '0')} / 09`;
      if (frameChanged && image.animate) {
        if (imageAnimation) imageAnimation.cancel();
        imageAnimation = image.animate([
          { opacity: .78, transform: 'scale(1.012)' },
          { opacity: 1, transform: 'scale(1)' }
        ], {
          duration: 280,
          easing: 'cubic-bezier(.22, 1, .36, 1)'
        });
      }

      filmstrip.querySelectorAll('.filmstrip-button').forEach((button, index) => {
        const selected = index + 1 === activeFrame;
        const distance = Math.abs(index + 1 - activeFrame);
        button.classList.toggle('active', selected);
        button.classList.toggle('wave-near', distance === 1);
        button.classList.toggle('wave-mid', distance === 2);
        button.classList.toggle('wave-far', distance >= 3);
        button.setAttribute('aria-selected', String(selected));
        if (selected && focusThumbnail) button.focus();
      });
    };

    const queueHoverFrame = frame => {
      pendingHoverFrame = frame;
      if (hoverFrameRequest) return;
      hoverFrameRequest = requestAnimationFrame(() => {
        hoverFrameRequest = 0;
        showFrame(pendingHoverFrame);
      });
    };

    for (let frame = 1; frame <= 9; frame += 1) {
      const button = document.createElement('button');
      button.className = 'filmstrip-button';
      button.type = 'button';
      button.setAttribute('role', 'option');
      button.setAttribute('aria-label', `Show frame ${frame}`);
      button.innerHTML = `<img src="static/images/cases/${caseKey}_f${frame}.jpg" alt="" loading="lazy" decoding="async">`;
      button.addEventListener('mouseenter', () => {
        if (hoverQuery.matches) queueHoverFrame(frame);
      });
      button.addEventListener('click', () => showFrame(frame));
      button.addEventListener('keydown', event => {
        if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
        event.preventDefault();
        showFrame(frame + (event.key === 'ArrowRight' ? 1 : -1), true);
      });
      filmstrip.appendChild(button);
    }

    showFrame(1);
  });
}

function initializeObservationAgents() {
  document.querySelectorAll('.agent-observation').forEach(root => {
    const study = root.closest('.evidence-study');
    const caseKey = study?.dataset.case;
    const observations = agentObservations[caseKey];
    if (!observations) return;

    const tabs = [...root.querySelectorAll('.observation-tab')];
    const panel = root.querySelector('.observation-panel');
    const title = root.querySelector('.analysis-title');
    const copy = root.querySelector('.analysis-copy');
    const counter = root.querySelector('.observation-counter');
    const panelId = `${caseKey}-observation-panel`;
    let activeIndex = 0;
    let animations = [];

    panel.id = panelId;
    tabs.forEach((button, index) => {
      button.id = `${caseKey}-${button.dataset.agent}-agent-tab`;
      button.setAttribute('aria-controls', panelId);

      button.addEventListener('click', () => showAgent(index));
      button.addEventListener('keydown', event => {
        let nextIndex = index;
        if (event.key === 'ArrowLeft') nextIndex = index - 1;
        else if (event.key === 'ArrowRight') nextIndex = index + 1;
        else if (event.key === 'Home') nextIndex = 0;
        else if (event.key === 'End') nextIndex = tabs.length - 1;
        else return;

        event.preventDefault();
        showAgent(nextIndex, true);
      });
    });

    function showAgent(index, focusTab = false) {
      activeIndex = (index + observationOrder.length) % observationOrder.length;
      const agentKey = observationOrder[activeIndex];
      const observation = observations[agentKey];
      title.textContent = observation.title;
      copy.textContent = observation.output;
      counter.textContent = `${String(activeIndex + 1).padStart(2, '0')} / 04`;

      tabs.forEach((button, buttonIndex) => {
        const selected = buttonIndex === activeIndex;
        button.classList.toggle('active', selected);
        button.setAttribute('aria-selected', String(selected));
        button.tabIndex = selected ? 0 : -1;
      });
      panel.setAttribute('aria-labelledby', tabs[activeIndex].id);

      if (title.animate) {
        animations.forEach(animation => animation.cancel());
        animations = [title, copy].map((element, elementIndex) => element.animate([
          { opacity: .25, transform: 'translate3d(8px, 0, 0)' },
          { opacity: 1, transform: 'translate3d(0, 0, 0)' }
        ], {
          duration: 320,
          delay: elementIndex * 28,
          easing: 'cubic-bezier(.22, 1, .36, 1)'
        }));
      }

      if (focusTab) tabs[activeIndex].focus();
    }

    showAgent(0);
  });
}

function initializeOutputPopover() {
  const popover = document.getElementById('output-popover');
  if (!popover) return;

  const triggers = [...document.querySelectorAll('.model-output-card[data-output-key]')];
  const title = popover.querySelector('#output-popover-title');
  const source = popover.querySelector('.output-popover-source');
  const status = popover.querySelector('.output-popover-status');
  const copy = popover.querySelector('.output-popover-copy');
  const closeButton = popover.querySelector('.output-popover-close');
  const hoverQuery = window.matchMedia('(hover: hover) and (pointer: fine)');
  let activeTrigger = null;
  let hideTimer = 0;
  let pinned = false;

  triggers.forEach(trigger => {
    trigger.setAttribute('role', 'button');
    const hint = trigger.querySelector('.output-hover-hint');
    if (hint?.firstChild) hint.firstChild.textContent = hoverQuery.matches ? 'Hover for full output ' : 'Tap for full output ';
  });

  const buildStatus = modelOutput => {
    status.replaceChildren();

    const verdict = document.createElement('span');
    verdict.className = `verdict-pill verdict-${modelOutput.verdict.toLowerCase()}`;
    verdict.textContent = modelOutput.verdict;

    const accuracy = document.createElement('span');
    accuracy.className = `accuracy-pill ${modelOutput.correct ? 'is-correct' : 'is-incorrect'}`;
    accuracy.textContent = modelOutput.correct ? 'Correct' : 'Incorrect';

    status.append(verdict, accuracy);
  };

  const clearHideTimer = () => {
    window.clearTimeout(hideTimer);
    hideTimer = 0;
  };

  const positionPopover = trigger => {
    popover.classList.remove('opens-left', 'opens-right');
    popover.style.left = '0px';
    popover.style.top = '0px';

    const triggerRect = trigger.getBoundingClientRect();
    const popoverRect = popover.getBoundingClientRect();
    const gutter = 16;
    const spacing = 14;
    const rightCandidate = triggerRect.right + spacing;
    const leftCandidate = triggerRect.left - popoverRect.width - spacing;
    const canOpenRight = rightCandidate + popoverRect.width <= window.innerWidth - gutter;
    let left;

    if (canOpenRight) {
      left = rightCandidate;
      popover.classList.add('opens-right');
    } else {
      left = Math.max(gutter, leftCandidate);
      popover.classList.add('opens-left');
    }

    const centeredTop = triggerRect.top + (triggerRect.height - popoverRect.height) / 2;
    const top = Math.min(
      Math.max(gutter, centeredTop),
      Math.max(gutter, window.innerHeight - popoverRect.height - gutter)
    );

    popover.style.left = `${Math.round(left)}px`;
    popover.style.top = `${Math.round(top)}px`;
  };

  const showPopover = (trigger, shouldPin = false) => {
    const modelOutput = modelOutputs[trigger.dataset.outputKey];
    if (!modelOutput) return;

    clearHideTimer();
    if (activeTrigger && activeTrigger !== trigger) activeTrigger.setAttribute('aria-expanded', 'false');
    activeTrigger = trigger;
    pinned = shouldPin;
    title.textContent = modelOutput.title;
    source.textContent = modelOutput.source;
    buildStatus(modelOutput);
    copy.textContent = modelOutput.output
      .replace(/\s*<answer>\s*(?:real|fake)\s*<\/answer>\s*$/i, '')
      .trim();
    copy.scrollTop = 0;
    popover.setAttribute('aria-hidden', 'false');
    popover.classList.add('is-visible');
    trigger.setAttribute('aria-expanded', 'true');

    const useModalLayout = !hoverQuery.matches || window.innerWidth <= 768;
    popover.classList.toggle('is-modal-layout', useModalLayout);
    document.body.classList.toggle('output-popover-open', useModalLayout);
    if (!useModalLayout) positionPopover(trigger);
  };

  const hidePopover = (force = false) => {
    clearHideTimer();
    if (pinned && !force) return;
    if (activeTrigger) activeTrigger.setAttribute('aria-expanded', 'false');
    activeTrigger = null;
    pinned = false;
    popover.classList.remove('is-visible', 'is-modal-layout', 'opens-left', 'opens-right');
    popover.setAttribute('aria-hidden', 'true');
    popover.style.removeProperty('left');
    popover.style.removeProperty('top');
    document.body.classList.remove('output-popover-open');
  };

  const scheduleHide = () => {
    clearHideTimer();
    hideTimer = window.setTimeout(() => hidePopover(), 130);
  };

  triggers.forEach(trigger => {
    trigger.addEventListener('mouseenter', () => {
      if (hoverQuery.matches) showPopover(trigger);
    });
    trigger.addEventListener('mouseleave', () => {
      if (hoverQuery.matches) scheduleHide();
    });
    trigger.addEventListener('focus', () => showPopover(trigger));
    trigger.addEventListener('blur', event => {
      if (!popover.contains(event.relatedTarget)) scheduleHide();
    });
    trigger.addEventListener('click', () => {
      if (hoverQuery.matches) return;
      if (activeTrigger === trigger && pinned) hidePopover(true);
      else showPopover(trigger, true);
    });
    trigger.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      showPopover(trigger, true);
      closeButton.focus();
    });
  });

  popover.addEventListener('mouseenter', clearHideTimer);
  popover.addEventListener('mouseleave', () => {
    if (hoverQuery.matches) scheduleHide();
  });
  closeButton.addEventListener('click', () => {
    const trigger = activeTrigger;
    hidePopover(true);
    if (trigger) trigger.focus();
  });
  document.addEventListener('pointerdown', event => {
    if (!pinned || popover.contains(event.target) || activeTrigger?.contains(event.target)) return;
    hidePopover(true);
  });
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape' || popover.getAttribute('aria-hidden') === 'true') return;
    const trigger = activeTrigger;
    hidePopover(true);
    if (trigger) trigger.focus();
  });
  window.addEventListener('scroll', () => hidePopover(true), { passive: true });
  window.addEventListener('resize', () => hidePopover(true));
}

function buildPromptLibrary() {
  const nav = document.getElementById('prompt-nav');
  const select = document.getElementById('prompt-select');
  const title = document.querySelector('.prompt-viewer-title');
  const group = document.querySelector('.prompt-viewer-group');
  const code = document.querySelector('.prompt-code');
  const copyButton = document.querySelector('.prompt-viewer .copy-prompt');
  let activeTemplate = 0;

  const showTemplate = index => {
    activeTemplate = index;
    const template = promptTemplates[index];
    title.textContent = template.title;
    group.textContent = `${template.group} · template ${String(index + 1).padStart(2, '0')}`;
    code.textContent = template.text;
    code.scrollTop = 0;
    code.classList.remove('template-enter');
    void code.offsetWidth;
    code.classList.add('template-enter');
    select.value = String(index);

    nav.querySelectorAll('.prompt-tab').forEach((button, buttonIndex) => {
      const selected = buttonIndex === index;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
  };

  promptTemplates.forEach((template, index) => {
    const button = document.createElement('button');
    button.className = 'prompt-tab';
    button.type = 'button';
    button.setAttribute('role', 'tab');
    button.innerHTML = `<strong>${template.title}</strong><small>${template.group}</small>`;
    button.addEventListener('click', () => showTemplate(index));
    button.addEventListener('keydown', event => {
      if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
      event.preventDefault();
      const next = (index + (event.key === 'ArrowDown' ? 1 : -1) + promptTemplates.length) % promptTemplates.length;
      showTemplate(next);
      nav.children[next].focus();
    });
    nav.appendChild(button);

    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${template.group} · ${template.title}`;
    select.appendChild(option);
  });

  select.addEventListener('change', () => showTemplate(Number(select.value)));
  copyButton.addEventListener('click', () => copyText(promptTemplates[activeTemplate].text, copyButton, 'Copy prompt'));
  showTemplate(0);
}

function initializeRevealEffects() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const targets = [...document.querySelectorAll('main > section:not(.publication-header)')];
  targets.forEach(target => target.classList.add('reveal-on-scroll'));
  document.documentElement.classList.add('reveal-ready');

  if (!('IntersectionObserver' in window)) {
    targets.forEach(target => target.classList.add('is-visible'));
    return;
  }

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    });
  }, { threshold: .08, rootMargin: '0px 0px -14% 0px' });

  targets.forEach(target => observer.observe(target));
}

document.addEventListener('DOMContentLoaded', () => {
  initializeCases();
  initializeObservationAgents();
  initializeOutputPopover();
  buildPromptLibrary();
  initializeRevealEffects();

  const bibtexButton = document.getElementById('copy-bibtex');
  bibtexButton.addEventListener('click', () => {
    const citation = document.getElementById('bibtex-code').textContent.trim();
    copyText(citation, bibtexButton, 'Copy');
  });
});
