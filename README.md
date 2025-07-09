# LiveKit Assistant

[Github](https://github.com/svpino/livekit-assistant)
[Youtube Video How to build a real-time AI assistant with voice and vision](https://www.youtube.com/watch?v=nvmV0a2geaQ)
[LiveKit Documentation](https://docs.livekit.io/agents/integrations/stt/deepgram/)


First, create a virtual environment, update pip, and install the required packages:

Please note that this agent requires specific versions of livekit agents and plugins.

That is why adding
```
pip install "livekit-agents[deepgram]~=1.0"
```
to .venv_livekit did not do the job.

```
$ python3 -m venv .venv_livekit_svpino
$ source .venv_svpino/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```

You need to set up the following environment variables:

```
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
```
Please also note that the code on the github repo has been updated and is different from the one in the video.

Then, run the assistant:

```
$ python3 assistant.py download-files
$ python3 assistant.py start
```

Finally, you can load the [hosted playground](https://agents-playground.livekit.io/) and connect it.