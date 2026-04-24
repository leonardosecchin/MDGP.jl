function check_param(cond, msg)
    if !cond
        throw(ArgumentError(msg))
    end
end

function check(cond, msg)
    if !cond
        throw(msg)
    end
end