--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local optnet = require 'optnet'

local checkpoint = {}

local function optimize(opt, net)
   if opt.shareGradInput then
      local sample_input = torch.randn(opt.batchSize,3,opt.dataset == 'imagenet' and 224 or 32):cuda()
      optnet.optimizeMemory(model, sample_input, {inplace = false, mode = 'training'})
   end
end

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))

   optimize(opt, latest)

   return latest, optimState
end

function checkpoint.save(epoch, model, optimState, bestModel)
   -- Don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   optnet.removeOptimization(model)

   local modelFile = 'model_' .. epoch .. '.t7'
   local optimFile = 'optimState_' .. epoch .. '.t7'

   torch.save(modelFile, model)
   torch.save(optimFile, optimState)
   torch.save('latest.t7', {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
   })

   if bestModel then
      torch.save('model_best.t7', model)
   end

   optimize(model)
end

return checkpoint
