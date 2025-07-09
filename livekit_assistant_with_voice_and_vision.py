#!/usr/bin/env python
# coding: utf-8

# In[1]:


# livekit_assistant_with_voice_and_vision (Livekit backend agent)


# In[ ]:


# Project folder: livekit-assistant-with-voice-and-vision

# readme.md

# [Github](https://github.com/svpino/livekit-assistant)
# [Youtube Video How to build a real-time AI assistant with voice and vision](https://www.youtube.com/watch?v=nvmV0a2geaQ)
# [LiveKit Documentation](https://docs.livekit.io/agents/integrations/stt/deepgram/)


# In[ ]:


# Run this Livekit agent
# Choose virtual environment .venv_livekit_svpino
# cd livekit-assistant-with-voice-and-vision
# copy .env from jupyter_notebook to livekit-assistant-with-voice-and-vision

# python livekit_assistant_with_voice_and_vision.py download-files
# python livekit_assistant_with_voice_and_vision.py start

# assistant.py is the original file from the github repository

# Finally, you can load the [hosted playground](https://agents-playground.livekit.io/) and connect it.


# In[ ]:


# choose kernel "Python (.venv_livekit_svpino)"


# In[ ]:


# First, create a virtual environment, update pip, and install the required packages:
# Please note that this agent requires specific versions of livekit agents and plugins.

# That is why adding
# $ pip install "livekit-agents[deepgram]~=1.0"
# to .venv_livekit did not do the job.

# $ python -m venv .venv_livekit_svpino
# $ source .venv_svpino/bin/activate
# $ pip install -U pip
# $ pip install -r requirements.txt

# Please also note that the code on the github repo has been updated and is different from the one in the video.

# pip install ipykernel
# pip freeze > .req_venv_livekit_svpino
# python -m ipykernel install --user --name=.venv_livekit_svpino --display-name "Python (.venv_livekit_svpino)"
# pip freeze > .req_venv_livekit_svpino


# In[ ]:


# .env

# DEEPGRAM_API_KEY=...

# copy .env from jupyter_notebook to livekit-assistant-with-voice-and-vision


# In[ ]:


import asyncio
from typing import Annotated
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

load_dotenv()

class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

    @agents.llm.ai_callable(
        description=(
            "Called when asked to evaluate something that would require vision capabilities,"
            "for example, an image, video, or the webcam feed."
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        return None


async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""

    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Alloy. You are a funny, witty bot. Your interface with users will be voice and vision."
                    "Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o")

    # Since OpenAI does not support streaming TTS, we'll use it with a StreamAdapter
    # to make it compatible with the VoiceAssistant
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: rtc.VideoFrame | None = None

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),  # We'll use Silero's Voice Activity Detector (VAD)
        stt=deepgram.STT(),  # We'll use Deepgram's Speech To Text (STT)
        llm=gpt,
        tts=openai_tts,  # We'll use OpenAI's Text To Speech (TTS)
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

#    chat = rtc.ChatManager(ctx.room)
    chat = assistant.start(ctx.room)

    async def _answer(text: str, use_image: bool = False):
        """
        Answer the user's message with the given text and optionally the latest
        image captured from the video track.
        """
        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """This event triggers whenever we get a new message from the user."""

        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""

        if len(called_functions) == 0:
            return

        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hi there! How can I help?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            # We'll continually grab the latest image from the video track
            # and store it in a variable.
            latest_image = event.frame


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

