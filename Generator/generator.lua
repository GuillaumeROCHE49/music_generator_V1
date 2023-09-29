function getClientInfo()
    return {
        name = "Music generator",
        category = "generator",
        author = "Guillaume ROCHE",
        versionNumber = 1,
        minEditorVersion = 65536
    }
end

function main()
    local generatorDir = "C:\\Users\\groch\\OneDrive\\Documents\\Dreamtonics\\Generator_Synth"  -- Can be change for your own path
    local pythonPath = generatorDir .. "\\venv\\Scripts\\python.exe"
    local generatorFile = generatorDir .. "\\music_gen\\py\\generate_all.py"

    ref = os.date("*t")
    date = ("%02d-%02d-%02d"):format(ref.year, ref.month, ref.day)
    time = ("%02d-%02d-%02d"):format(ref.hour, ref.min, ref.sec)
    local outputDir = generatorDir .. "\\music_gen\\output\\" .. date .. "_" .. time;
    os.execute("mkdir " .. outputDir)
    local prompt = SV:showInputBox("Music generator", "Enter your prompt, please.")
    if prompt == "" then
        SV:showMessageBox("Music generator", "Error: prompt is empty")
        return
    end
    local step = SV:showInputBox("Music generator", "Enter the number of steps, please... Then wait!")
    if step == "" then
        SV:showMessageBox("Music generator", "Error: step is empty")
        return
    end

    local commande = pythonPath .. ' ' .. generatorFile .. ' -env ' .. pythonPath .. ' -o ' .. outputDir .. ' -p "' .. prompt .. '" -s '.. step .. ' -k "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"'    
    -- Commande : C:\Users\groch\OneDrive\Documents\Dreamtonics\Generator_Synth\venv\Scripts\python.exe C:\Users\groch\OneDrive\Documents\Dreamtonics\Generator_Synth\music_gen\py\generate_all.py -env C:\Users\groch\OneDrive\Documents\Dreamtonics\Generator_Synth\venv\Scripts\python.exe -o C:\Users\groch\OneDrive\Documents\Dreamtonics\Generator_Synth\music_gen\output\2021-04-25_22-00-00 -p "Lofi 80's music with smooth drum" -s 70 -k "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"

    -- error file
    local errorFile = outputDir .. "\\log.txt"
    local result = executeCommand(commande, errorFile)
    
    local fileToVerify = {outputDir .. "\\music.wav", outputDir .. "\\music.txt", outputDir .. "\\music.mid.txt"}
    for i, file in ipairs(fileToVerify) do
        if verifyFile(file) then
            SV:showMessageBox("Music generator", "Error: " .. file .. " not found")
            return
        end
    end

    -- Read the .mid.txt file and create the notes
    local notes = {}
    local f = io.open(outputDir .. "\\music.mid.txt", "r")

    local lines = f:read("*line")

    -- line look like this : 0;0;2216;0.7142857142857143;0.7142857142857143;100;In
    -- track, channel, frequency, start, duration, velocity, lyric
    for line in lines:gmatch("[^\r\n]+") do
        local note = SV:create("Note")
        
        -- split the line with ;
        local values = {}
        for value in line:gmatch("[^;]+") do
            values[#values + 1] = value
        end
        SV:showInputBox("Music generator", "Number of values : " .. #values)
        if #values  == 7 then
            SV:showInputBox("Music generator", "Pitch : " .. SV:freq2Pitch(tonumber(values[3])) .. " start : " .. values[4] .. " duration : " .. values[5] .. " lyric : " .. values[7])
            -- convert frequency to pitch
            note:setPitch(SV:freq2Pitch(tonumber(values[3])))
            note:setOnset(tonumber(values[4]))
            note:setDuration(tonumber(values[5]))
            note:setLyrics(values[7])
            notes[#notes + 1] = note
        end
    end
    f:close()

    SV:showInputBox("Music generator", "Number of notes : " .. #notes)

    -- Create the note group
    local noteGroup = SV:create("NoteGroup")
    noteGroup:setName("Generated lyrics")
    for note in notes do
        noteGroup:addNote(note)
    end
    SV:getProject():addNoteGroup(noteGroup)

    SV:finish()
end

function executeCommand(command, errorFile)
    -- Execute command using os.execute
    local cmd = command .. " 2> " .. errorFile
    local result = os.execute(cmd)
    return result
end

function verifyFile(file)
    local f = io.open(file, "r")
    if f then
        io.close(f)
        return false
    else
        return true
    end
end