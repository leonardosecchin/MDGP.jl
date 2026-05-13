function check_param(cond, msg)
    if !cond
        throw(ArgumentError(msg))
    end
end

function check(cond, msg)
    if !cond
        error(msg)
    end
end